"""Small, strict LaTeX-math subset to native MathML. Python standard library only.

Not TeX: no macros, files, URLs, HTML, packages, implicit command fallback or execution.
8192 source characters, 4096 tokens, 48 nesting levels. Errors carry source offsets.
"""
import re
import unicodedata
import xml.etree.ElementTree as ET

SYMBOLS = dict(zip(
    'alpha beta gamma delta epsilon theta lambda mu nu xi pi rho sigma tau phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega'.split(),
    'α β γ δ ε θ λ μ ν ξ π ρ σ τ φ χ ψ ω Γ Δ Θ Λ Ξ Π Σ Φ Ψ Ω'.split()))
OPERATORS = dict(zip('cdot times le leq ge geq ne neq approx in to mapsto partial nabla infty pm cup cap'.split(),
                     '· × ≤ ≤ ≥ ≥ ≠ ≠ ≈ ∈ → ↦ ∂ ∇ ∞ ± ∪ ∩'.split()))

def el(tag, text=None, *children, **attrs):
    result=ET.Element(tag,attrs)
    result.text=text
    result.extend(children)
    return result

class Parser:
    def __init__(self, source):
        if not isinstance(source,str) or not source.strip() or len(source)>8192:
            raise ValueError('math source must contain 1–8192 characters')
        self.source=source
        self.tokens=list(re.finditer(r'\\[a-zA-Z]+|\\.|[0-9]+(?:\.[0-9]+)?|[^\s]',source))
        if len(self.tokens)>4096:raise ValueError('math token limit exceeded')
        self.pos=0;self.depth=0;self.atom_depth=0

    def peek(self):
        return self.tokens[self.pos].group() if self.pos<len(self.tokens) else ''

    def error(self, message):
        offset=self.tokens[self.pos].start() if self.pos<len(self.tokens) else len(self.source)
        raise ValueError(f'{message} at character {offset}')

    def take(self, expected=None):
        value=self.peek()
        if not value or (expected is not None and value!=expected):self.error('Expected '+str(expected))
        self.pos+=1
        return value

    def raw_group(self):
        self.take('{');start=self.tokens[self.pos-1].end()
        while self.peek() and self.peek()!='}':
            if self.peek()=='{':self.error('Nested text groups unsupported')
            self.take()
        end=self.tokens[self.pos].start() if self.peek() else len(self.source)
        self.take('}')
        return self.source[start:end]

    def group(self):
        self.take('{');value=self.row({'}'});self.take('}')
        return value

    def row(self, stops=frozenset()):
        self.depth+=1
        if self.depth>48:self.error('Nesting limit exceeded')
        children=[]
        while self.peek() and self.peek() not in stops:
            base=self.atom();scripts={}
            while self.peek() in ('_','^'):
                op=self.take()
                if op in scripts:self.error('Duplicate script')
                scripts[op]=self.group() if self.peek()=='{' else self.atom()
            if scripts:
                under=base.tag=='mo' and base.text in ('∑','∏')
                if len(scripts)==2:
                    base=el('munderover' if under else 'msubsup',None,base,scripts['_'],scripts['^'])
                else:
                    op=next(iter(scripts));tag=('munder' if op=='_' else 'mover') if under else ('msub' if op=='_' else 'msup')
                    base=el(tag,None,base,scripts[op])
            children.append(base)
        self.depth-=1
        return el('mrow',None,*children)

    def atom(self):
        self.atom_depth+=1
        try:
            if self.atom_depth>48:self.error('Nesting limit exceeded')
            return self._atom()
        finally:self.atom_depth-=1

    def _atom(self):
        value=self.peek()
        if value=='{':return self.group()
        self.take()
        if value in ('}', '^','_','&',r'\\',r'\end',r'\right'):
            self.error('Unexpected '+value)
        if value==r'\frac':return el('mfrac',None,self.group(),self.group())
        if value==r'\sqrt':return el('msqrt',None,self.group())
        if value in (r'\text',r'\mathrm',r'\mathbf',r'\operatorname'):
            if value in (r'\text',r'\operatorname'):
                return el('mtext' if value==r'\text' else 'mo',self.raw_group())
            group=self.group()
            for token in group.iter():
                if token.tag not in ('mi','mn') or not token.text:continue
                if value==r'\mathrm':token.set('mathvariant','normal')
                else:
                    try:
                        token.text=''.join(unicodedata.lookup('MATHEMATICAL BOLD '+unicodedata.name(c).replace('LATIN ','').replace('GREEK ','').replace('LETTER ','')) for c in token.text)
                    except (KeyError,ValueError):self.error('Unsupported bold identifier')
            return group
        if value in (r'\vec',r'\hat',r'\bar'):
            return el('mover',None,self.group() if self.peek()=='{' else self.atom(),el('mo',{r'\vec':'→',r'\hat':'^',r'\bar':'¯'}[value]),accent='true')
        if value==r'\left':
            left=self.take();allowed={'(':')','[':']',r'\{':r'\}','|':'|','.':'.'}
            if left not in allowed:self.error('Unsupported left delimiter')
            body=self.row({r'\right'});self.take(r'\right');right=self.take()
            if right not in (allowed[left],'.') and left!='.':self.error('Mismatched delimiters')
            if right not in (')',']',r'\}','|','.'):self.error('Unsupported right delimiter')
            return el('mrow',None,el('mo','' if left=='.' else left.lstrip('\\'),stretchy='true'),body,
                      el('mo','' if right=='.' else right.lstrip('\\'),stretchy='true'))
        if value==r'\begin':return self.environment()
        if value in (r'\,',r'\;',r'\!',r'\quad',r'\qquad'):
            return el('mspace',width={r'\,':'0.167em',r'\;':'0.278em',r'\!':'-0.167em',r'\quad':'1em',r'\qquad':'2em'}[value])
        if value.startswith('\\'):
            command=value[1:]
            if command in SYMBOLS:return el('mi',SYMBOLS[command])
            if command in OPERATORS:return el('mo',OPERATORS[command])
            if command in ('sum','prod','int'):return el('mo',{'sum':'∑','prod':'∏','int':'∫'}[command],largeop='true')
            if command in ('sin','cos','tan','log','exp','min','max','lim'):return el('mi',command,mathvariant='normal')
            if command in ('{','}','%','_','|'):return el('mo','‖' if command=='|' else command)
            self.error('Unsupported command '+value)
        if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',value):return el('mn',value)
        if value.isalpha():return el('mi',value)
        if value in '+-=()[]|/,.:<>!':return el('mo',value)
        self.error('Unsupported character '+value)

    def environment(self):
        name=self.raw_group()
        if name not in ('matrix','pmatrix','bmatrix','aligned','cases'):self.error('Unsupported environment '+name)
        rows=[];width=None
        while True:
            cells=[el('mtd',None,self.row({'&',r'\\',r'\end'}))]
            while self.peek()=='&':
                self.take();cells.append(el('mtd',None,self.row({'&',r'\\',r'\end'})))
            if width is not None and width!=len(cells):self.error('Ragged matrix')
            width=len(cells);rows.append(el('mtr',None,*cells))
            if self.peek()==r'\end':break
            self.take(r'\\')
        self.take(r'\end')
        if self.raw_group()!=name:self.error('Mismatched environment')
        table=el('mtable',None,*rows,columnalign='right left' if name=='aligned' else 'left' if name=='cases' else 'center')
        if name in ('pmatrix','bmatrix','cases'):
            left,right={'pmatrix':('(',')'),'bmatrix':('[',']'),'cases':('{','')}[name]
            return el('mrow',None,el('mo',left,stretchy='true'),table,el('mo',right,stretchy='true'))
        return table

def render(source, *, display=True):
    parser=Parser(source)
    root=el('math',None,el('semantics',None,parser.row(),el('annotation',source,encoding='application/x-tex')),
            xmlns='http://www.w3.org/1998/Math/MathML',display='block' if display else 'inline')
    return ET.tostring(root,encoding='unicode',short_empty_elements=False)
