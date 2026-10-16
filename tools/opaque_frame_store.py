"""Development-only bounded disk frame/cotangent store; never a render certificate.
No mmap, full-frame tensors, overwrite, missing-pixel completion or implicit replay.
The supplied identity and manifest hash must come from the supervising receipt.
"""
from pathlib import Path
from contextlib import ExitStack
import hashlib,json,os

CHANNELS={'colour':3,'mask':1,'gcolour':3,'gmask':1}
MAX_BLOCK_PIXELS=512
MAX_PIXELS=1920*1080
MAX_MANIFEST_BYTES=1048576

def _digest(parts):
    h=hashlib.sha256()
    for k in CHANNELS:h.update(parts[k])
    return h.hexdigest()

def _extent(width,height,block):
    if any(type(v) is not int or v<1 for v in (width,height,block)) or width*height>MAX_PIXELS or block>MAX_BLOCK_PIXELS:raise ValueError('finite frame/block extent')

def _encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class Writer:
    def __init__(self,root,width,height,identity,block_pixels=512):
        _extent(width,height,block_pixels);self.identity=json.loads(_encode(identity));self.root=Path(root);self.root.mkdir()
        self.width=width;self.height=height;self.cap=block_pixels;self.next_pixel=0;self.blocks=[];self.closed=False
        self.paths={k:self.root/(k+'.f64') for k in CHANNELS};self.files={k:p.open('xb',buffering=0) for k,p in self.paths.items()};self.hashes={k:hashlib.sha256() for k in CHANNELS}
    def append(self,offset,count,values):
        if self.closed or type(offset) is not int or offset!=self.next_pixel or type(count) is not int or not 1<=count<=self.cap or offset+count>self.width*self.height:raise ValueError('ordered bounded frame append required')
        if set(values)!=set(CHANNELS) or any(type(values[k]) is not bytes or len(values[k])!=count*n*8 for k,n in CHANNELS.items()):raise ValueError('exact float64 byte lengths required')
        # Every validation precedes writes. I/O failure leaves an incomplete store.
        for k,f in self.files.items():
            if f.write(values[k])!=len(values[k]):raise OSError('short frame write')
            self.hashes[k].update(values[k])
        self.blocks.append(dict(offset=offset,count=count,sha256=_digest(values)));self.next_pixel+=count
    def close(self):
        if not self.closed:
            for f in self.files.values():f.close()
            self.closed=True
    def finish(self):
        if self.closed or self.next_pixel!=self.width*self.height:raise ValueError('incomplete frame cannot be sealed')
        row=dict(version=1,dtype='<f8',width=self.width,height=self.height,block_pixels=self.cap,identity=self.identity,blocks=self.blocks,streams={k:dict(bytes=self.next_pixel*n*8,sha256=self.hashes[k].hexdigest()) for k,n in CHANNELS.items()})
        raw=_encode(row)
        if len(raw)>MAX_MANIFEST_BYTES:raise ValueError('frame manifest byte ceiling')
        for f in self.files.values():f.flush();os.fsync(f.fileno())
        self.close()
        with (self.root/'frame.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        return dict(root=self.root.as_posix(),manifest_sha256=hashlib.sha256(raw).hexdigest(),manifest_bytes=len(raw),pixels=self.next_pixel,storage_bytes=sum(x['bytes'] for x in row['streams'].values()),payload_limit_bytes=self.cap*64)

class Reader:
    def __init__(self,root,manifest_sha256,identity):
        self.root=Path(root)
        with (self.root/'frame.json').open('rb') as f:raw=f.read(MAX_MANIFEST_BYTES+1)
        if len(raw)>MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('frame manifest hash/size')
        r=json.loads(raw);self.manifest=r
        if r.get('version')!=1 or r.get('dtype')!='<f8' or r.get('identity')!=identity:raise ValueError('frame identity/format mismatch')
        _extent(r['width'],r['height'],r['block_pixels']);next_pixel=0
        for b in r['blocks']:
            n=b['count']
            if type(n) is not int or not 1<=n<=r['block_pixels'] or b['offset']!=next_pixel:raise ValueError('frame block order/extent')
            next_pixel+=n
        if next_pixel!=r['width']*r['height'] or set(r['streams'])!=set(CHANNELS):raise ValueError('incomplete frame metadata')
        for k,n in CHANNELS.items():
            p=self.root/(k+'.f64');expected=next_pixel*n*8
            if p.is_symlink() or p.stat().st_size!=expected or r['streams'][k]['bytes']!=expected:raise ValueError('frame stream length/type')
    def blocks(self):
        # Each block is verified before exposing its bytes. A complete replay must
        # consume the iterator, whose final stream hashes are also checked.
        with ExitStack() as stack:
            files={k:stack.enter_context((self.root/(k+'.f64')).open('rb',buffering=0)) for k in CHANNELS};hashes={k:hashlib.sha256() for k in CHANNELS}
            for b in self.manifest['blocks']:
                values={k:f.read(b['count']*CHANNELS[k]*8) for k,f in files.items()}
                if any(len(values[k])!=b['count']*n*8 for k,n in CHANNELS.items()) or _digest(values)!=b['sha256']:raise ValueError('frame block hash/length differs')
                for k,v in values.items():hashes[k].update(v)
                yield b['offset'],b['count'],values
            if any(f.read(1) for f in files.values()) or any(hashes[k].hexdigest()!=self.manifest['streams'][k]['sha256'] for k in CHANNELS):raise ValueError('frame final stream hash/length differs')
