"""Build/check the authored mini-textbook. No third-party packages or network access."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from .mathml import render
from .textbook import CHAPTERS

ROOT=Path(__file__).parent
OUTPUT=ROOT/'static/learn/energy-conservation.json'

def asset_name(identifier):
    names={c['id']:c['id'].rsplit('/',1)[-1]+'.json' for c in CHAPTERS}
    names['learn/index']='index.json'
    return names[identifier]  # Explicit registry; never a caller-provided filesystem path.

def fingerprint():
    return hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in
        ('build_lessons.py','mathml.py','textbook.py'))).hexdigest()

def link(identifier,title):
    return '<a href="/testing/?page='+html.escape(identifier,quote=True)+'">'+html.escape(title)+'</a>'

def navigation(identifier):
    ids=[c['id'] for c in CHAPTERS];i=ids.index(identifier)
    links=[link('learn/index','Textbook contents')]
    if i:links.append(link(ids[i-1],'Previous: '+CHAPTERS[i-1]['title']))
    if i+1<len(ids):links.append(link(ids[i+1],'Next: '+CHAPTERS[i+1]['title']))
    return '<nav class="book-navigation" aria-label="Chapter navigation">'+' '.join(links)+'</nav>'

def compiled(body):
    return re.sub(r'@@(.*?)@@(.*?)@@',lambda match:equation(*match.groups()),body,flags=re.S)

def articles():
    result={}
    titles={c['id']:c['title'] for c in CHAPTERS}
    for c in CHAPTERS:
        if c['body'] is None:
            result[c['id']]=article();continue
        prereqs=', '.join(link(p,titles[p]) for p in c['prerequisites']) or 'No previous chapter required.'
        evidence=''.join('<li>'+link(p,p)+'</li>' for p in c['evidence'])
        body=('<article class="lesson">'+navigation(c['id'])+
            '<p class="lesson-notice">Original learning chapter · updated 7 September 2026. '
            'Equations and examples explain concepts; they are not test acceptance or a support matrix.</p>'+
            '<p>Prerequisites: '+prereqs+'</p>'+compiled(c['body'])+
            '<h2>Follow the evidence</h2><ul>'+evidence+'</ul><p>Implementation or fixture source: <code>'+
            html.escape(c['source'])+'</code>. Check the current tests for exact supported behavior.</p>'+
            navigation(c['id'])+'</article>')
        result[c['id']]={'id':c['id'],'updated_on':'2026-09-07','source_sha256':fingerprint(),'html':body}
    contents=''.join('<li>'+link(c['id'],c['title'])+'</li>' for c in CHAPTERS)
    body=('<article class="lesson"><p class="lesson-notice">A growing, original mini-textbook for gffx users. '
        'Start with the foundations or follow a test page back to its mathematics. This is not a claim that every topic is implemented.</p>'+
        '<h2>Read in this order</h2><ol>'+contents+'</ol>'+
        '<h2>How to use this book</h2><p>Each chapter connects intuition, notation, a worked example, '
        'an exercise with an answer, implementation boundaries and live evidence. Prerequisites and previous/next '
        'links provide a learning path. Examples are small enough to check by hand. The energy chapter preserves '
        'the rejected model and its dated measurements instead of rewriting our history.</p>'+
        '<h2>What exists and what is illustrative</h2><p>Baseline mesh, transform, proximity, sampling, rasterization '
        'and texture primitives and the eager camera layer have implementation fixtures. Material energy is currently '
        'an independent mathematical reference, not a public shading API. Saved animations and the ball are demonstrations, '
        'not the planned animation or simulation system. No chapter expands platform support or closes a milestone.</p>'+
        '<h2>Future chapters</h2><p>Lights and spherical harmonics now extend the energy chapter. '
        'Continue alongside accepted work: scene organization, differentiable material budgets, skinning and blendshapes, deferred composition, acceleration '
        'structures, visibility derivatives, color management, path tracing, volumes and physical simulation. '
        'These are a curriculum backlog, not implemented or completed features.</p>'+
        '<h2>Keep the textbook growing</h2><p>Every new graphics capability or changed mathematical contract needs '
        'a chapter or an update, a checked worked example, its assumptions and singularities, gradient limits, '
        'and links to tests and suitable finite demonstrations. Update source and generated pages together. '
        'Explanations never replace independent acceptance tests.</p>'+
        '<h2>Further reading</h2><p>These original notes complement detailed primary treatments: '
        '<a href="https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations">PBRT transformations</a>, '
        '<a href="https://pbr-book.org/4ed/Shapes/Triangle_Meshes">triangle meshes</a>, '
        '<a href="https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions">sampling</a> and '
        '<a href="https://pbr-book.org/4ed/Textures_and_Materials/Image_Texture">image textures</a>. '
        'Their conventions and implementation scopes are not automatically gffx conventions.</p></article>')
    result['learn/index']={'id':'learn/index','updated_on':'2026-09-07','source_sha256':fingerprint(),'html':body}
    return result

def equation(source, explanation):
    return ('<figure class="lesson-equation" tabindex="0" aria-label="Mathematical equation">'+render(source)+
            '<figcaption>'+html.escape(explanation)+'</figcaption><details><summary>Copyable LaTeX source</summary><pre><code>'+
            html.escape(source)+'</code></pre></details></figure>')

def article():
    body=r'''<article class="lesson">
<p class="lesson-notice">Learning article · authored 7 September 2026 · approximately 12 minutes.
This explains an independent mathematical reference, not a public gffx shading implementation.</p>
<nav aria-label="Article contents"><h2>In this article</h2><ol>
<li><a href="#energy-intuition">A surface has a light budget</a></li>
<li><a href="#energy-units">Radiance, irradiance and angles</a></li>
<li><a href="#energy-brdf">What a BRDF describes</a></li>
<li><a href="#energy-lambert">Why Lambert has a pi</a></li>
<li><a href="#energy-coupling">How we accidentally created light</a></li>
<li><a href="#energy-revision">The revised budget</a></li>
<li><a href="#energy-numerics">Testing the mathematics and the calculation</a></li>
<li><a href="#light-recipes">Point lights and distant lights</a></li>
<li><a href="#light-sh">Spherical harmonics and approximation error</a></li></ol></nav>
<section id="energy-intuition"><h2>1. A surface has a light budget</h2>
<p>Imagine sending 100 units of light energy onto a non-emitting surface. Some can leave as a sharp reflection,
some as a broad, matte reflection, and some can be absorbed. They cannot add up to 140 units.
We are describing steady reflection, not a lamp, fluorescence or stored energy being released.</p>
<figure><svg viewBox="0 0 640 120" role="img" aria-label="100 incoming units split into 30 glossy, 50 matte and 20 absorbed units">
<rect x="10" y="15" width="620" height="28" fill="#356fa1"></rect>
<rect x="10" y="60" width="186" height="28" fill="#a65c21"></rect>
<rect x="196" y="60" width="310" height="28" fill="#288375"></rect>
<rect x="506" y="60" width="124" height="28" fill="#647080"></rect>
<text x="10" y="112">30 glossy</text><text x="215" y="112">50 matte</text><text x="515" y="112">20 absorbed</text></svg>
<figcaption>Illustrative energy accounting: 100 in, 80 reflected, 20 absorbed. Not a measured material.</figcaption></figure>
<p>“Energy conserving” means the reflected total never exceeds the incoming total. It does not mean every
surface reflects everything. A dark surface is allowed to absorb most of its light. An individual bright
highlight is also allowed: concentration into a narrow direction is not the same as creating energy.</p></section>
<section id="energy-units"><h2>2. Radiance, irradiance and angles</h2>
<p><strong>Radiance</strong> describes directional light: how much travels along a direction per projected
area and per solid angle. Its units are watts per square metre per steradian. <strong>Irradiance</strong>
is all the arriving power per surface area, summed over directions, in watts per square metre.</p>
<p>A solid angle is an area on a unit sphere, measured in steradians. A hemisphere contains all directions
above a flat surface. The surface normal N is its perpendicular unit vector. L points from the surface
toward the incoming light; V points toward the viewer. These arrows must be in the same coordinate system.</p>
@@\mu = \mathbf{N} \cdot \mathbf{L} = \cos\theta@@Mu is the cosine of the angle between the normal N and light direction L. Both vectors have unit length.@@
<p>Facing light head-on gives a cosine of one. Tilting spreads the same beam over more surface area, so less
power lands per unit area. At a right angle the cosine is zero. For this discussion, directions below the
surface are excluded; this is separate from any two-sided material rule.</p>
@@I = \int_{\Omega^+} L_i(\mathbf{L})\, (\mathbf{N}\cdot\mathbf{L})\, d\omega@@I denotes irradiance in this article; Li is incident radiance, Omega+ is the upper hemisphere, and d-omega is a tiny solid-angle element.@@
<p>The integral means “add contributions from every incoming direction.” The cosine is part of the
physical projected-area weighting, not a brightness adjustment chosen for appearance.</p></section>
<section id="energy-brdf"><h2>3. A BRDF describes directional reflection</h2>
<p>A bidirectional reflectance distribution function, or BRDF, connects an incoming direction to an outgoing
direction. Write it as f(V,L). It has units of inverse steradians. It is not itself a percentage and may
be larger than one in a narrow highlight; the integrated reflected energy is what must be bounded.</p>
@@L_o(\mathbf{V}) = \int_{\Omega^+} f(\mathbf{V},\mathbf{L})\,L_i(\mathbf{L})\,(\mathbf{N}\cdot\mathbf{L})\,d\omega@@Lo is outgoing radiance. The BRDF f weights each incoming direction before all its contributions are added.@@
<p>For the reciprocal reflection models here, exchanging V and L leaves the BRDF unchanged. This is called
reciprocity. Checking it catches corrections that behave differently when light and viewer exchange places.</p>
@@f(\mathbf{V},\mathbf{L})=f(\mathbf{L},\mathbf{V})@@Reciprocity concerns the BRDF, not the final image or the light-source intensity.@@
<p>A <strong>white-furnace test</strong> places the material in a uniform environment with incident radiance
one. For a reciprocal non-emitting BRDF, the cosine-weighted hemisphere integral must not exceed one.
The test checks energy accounting, not whether a material resembles measured skin or a particular renderer.</p></section>
<section id="energy-lambert"><h2>4. Why Lambert includes 1/pi</h2>
<p>Lambert is an ideal matte reflection model. Its BRDF is independent of light and viewing direction.
Albedo A is the reflected fraction, between zero and one in each linear colour channel.</p>
@@f_d=\frac{A}{\pi}@@The subscript d means diffuse. A is albedo and pi is the usual mathematical constant.@@
<details><summary>Derivation: where does pi come from?</summary>
<p>Let theta run from the normal to the horizon and phi around the normal. The solid-angle element is
sin(theta) d-theta d-phi. Multiplying it by the projected-area cosine and integrating gives pi:</p>
@@\int_0^{2\pi}\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,d\phi=\pi@@There are 2pi radians around the normal; the polar integral contributes one half.@@
@@\int_{\Omega^+}\frac{A}{\pi}\,(\mathbf{N}\cdot\mathbf{L})\,d\omega=A@@Dividing the BRDF by pi makes its total reflected fraction equal to the requested albedo.@@
<p>Using A instead of A/pi with the same physical lighting units would return pi times too much energy.
A legacy renderer may absorb that factor into its light convention; mixing conventions is the error.</p></details></section>
<section id="energy-coupling"><h2>5. How we accidentally created light</h2>
<p>We also want a glossy component. Our proposed specular term uses GGX for the distribution of tiny surface
orientations, correlated Smith visibility for masking and shadowing, and Schlick Fresnel for angle-dependent
reflection. Call that whole specular BRDF s. F0 is its normal-incidence Fresnel reflectance, not an arbitrary
brightness multiplier. Perceptual roughness r maps to the GGX width alpha=r squared.</p>
@@H=\frac{V+L}{\|V+L\|}@@H is the unit halfway vector between outward unit view and light directions.@@
@@F=F_0+(1-F_0)(1-V\cdot H)^5@@Schlick Fresnel grows toward grazing microfacet incidence. F0 is between zero and one.@@
<p>Our first mixture attenuated Lambert by this local Fresnel value and then added s. That sounds plausible,
but the local Fresnel factor is not the total specular energy over the hemisphere.</p>
@@f_{old}=(1-F)\frac{A}{\pi}+s@@Rejected local-Fresnel mixture. This equation is kept as a negative control, not an accepted conserving model.@@
<h3>Worked example: a failed light budget</h3>
<p>At white albedo, F0=0.04, roughness=0.2 and view cosine=0.1, independent integration gave
<strong>1.42787</strong> out for one unit of incident environment radiance—about 42.8 percent too much.
The reference was refined before we attributed the failure to the model. We did not clamp the result
or increase the allowable energy threshold.</p></section>
<section id="energy-revision"><h2>6. Give both components the same budget</h2>
<p>First integrate the specular BRDF to determine its directional reflected fraction E(V).
Here E means a dimensionless energy fraction, not the irradiance I used earlier. Let E-average be its
cosine-weighted mean over the hemisphere, divided by pi. T(V)=1-E(V) is the remaining fraction.</p>
@@E(V)=\int_{\Omega^+}s(V,L)(N\cdot L)\,d\omega@@Directional specular reflectance: integrate the entire specular BRDF, not just Fresnel.@@
@@E_{avg}=\frac{1}{\pi}\int_{\Omega^+}E(L)(N\cdot L)\,d\omega@@The average has cosine weighting and is dimensionless.@@
@@f_{matte}(V,L)=\frac{A(1-E(V))(1-E(L))}{\pi(1-E_{avg})}@@The reciprocal coupling gives the matte term only the remaining specular budget. The mean in the denominator is part of the model, not an output clamp.@@
<details><summary>Proof: no double-counting, and reciprocity survives</summary>
<p>When integrating over L, A and E(V) are constant. The integral of (1-E(L)) times the cosine is
pi times (1-E-average), which cancels the denominator. The matte contribution is therefore A(1-E(V)).</p>
@@E_{total}(V)=E(V)+A(1-E(V))\leq 1@@For A and E(V) between zero and one, total reflection stays between zero and one. White A=1 returns one; black A=0 leaves only specular reflection.@@
<p>The product of the two residual factors is unchanged when V and L swap, so the coupling is reciprocal.
If the average residual is exactly zero, the matte term is defined as zero rather than evaluating 0/0.
This proof assumes exact integrals; approximate tables still need numerical tests.</p></details>
<section class="energy-playground" aria-label="Illustrative energy budget calculator">
<h3>Try the budget</h3><p>This simple calculator illustrates the accounting identity, not a live GGX renderer or a measured material.</p>
<label>Specular energy fraction E <input data-energy="specular" type="range" min="0" max="1" step="0.01" value="0.3"></label>
<label>Matte albedo A <input data-energy="albedo" type="range" min="0" max="1" step="0.01" value="0.5"></label>
<output data-energy="result" aria-live="polite">E=0.30; A=0.50; glossy 0.30 + matte 0.35 = 0.65 reflected; 0.35 absorbed.</output>
</section>
<p>With our revised formula, the original counterexample gives <strong>1.00003</strong> at the refined grid.
That tiny excess is within the declared numerical tolerance, not literal exact conservation by floating-point code.
The coupled matte model is physically plausible and empirical: it is not an exact coating simulation, a
Blender Principled replica, or a demonstrated skin-scattering model.</p></section>
<section id="energy-numerics"><h2>7. Test the calculation as well as the mathematics</h2>
<p>An integral is continuous; a computer samples it. A narrow highlight may fit between sample locations.
We therefore compare coarse and fine calculations, and use two different parameterizations for the specular
integral: direct light-hemisphere integration and visible-normal integration.</p>
<p>The declared sweep contains 108 combinations of roughness, F0, view angle and albedo. At the recorded
revision the maximum white-albedo error is 0.000225413; the maximum coarse/fine change is 0.000362298.
The upper energy bound remains 1.001 and the refinement change must stay below 0.001. These finite cases
do not prove every roughness, angle, device or implementation correct.</p>
<p>Gradients need their own checks. When fitting roughness or F0, their effects on E and E-average must
also be differentiated. Our current diagnostic detaches those integrals; it is <strong>not a public,
fully differentiable material implementation</strong>. The practical runtime evaluator is still to be designed.</p>
<h3>Follow the evidence</h3><ul>
<li><a href="/testing/?page=shading/material-contract">Material energy contract: current tests and their measurements</a></li>
<li>Authored source: tools/testing_dashboard/build_lessons.py. Reference math: tools/shading_energy_probe.py.</li>
<li>Tests: tests/python/test_shading_contract.py. Recorded measurements: build/shading-revision-20260907/revised-matrix.json, 7 September 2026.</li>
<li>Development source at measurement: e92ed673d883a8b84dc6564172650b2e82f45534 plus uncommitted reference changes. This article is a dated explanation; the linked tests may have newer results.</li></ul>
<h3>Primary reading</h3><ul>
<li><a href="https://cg.iit.bme.hu/~szirmay/scook.pdf">Kelemen and Szirmay-Kalos (2001), section 2.2: specular/matte coupling</a></li>
<li><a href="https://jcgt.org/published/0007/04/01/paper.pdf">Heitz (2018): sampling GGX visible normals</a></li>
<li><a href="https://pbr-book.org/4ed/Reflection_Models/BSDF_Representation">PBRT: BRDFs, reciprocity and energy conservation</a></li></ul>
<details><summary>Try it yourself</summary><p>If specular reflection uses 0.3 of the budget and matte albedo is 0.5, how much is reflected and how much is absorbed?</p><h3>Answer</h3><p>The remaining budget is 0.7. Matte reflection uses half of that, 0.35, so total reflection is 0.65 and absorption is 0.35. The calculator above checks this accounting identity, not a complete GGX integration.</p></details>
<p>Continue learning through the textbook contents and chapter navigation.</p></section></article>'''
    body=body.replace('</article>', lighting_section()+lobe_section()+budget_section()+material_section()+table_section()+sharp_section()+'</article>')
    body=compiled(body)
    nav=navigation('learn/shading/energy-conservation')
    body=body.replace('<article class="lesson">','<article class="lesson">'+nav).replace('</article>',nav+'</article>')
    # Only local authored text reaches this generated output. No Markdown/HTML from producers.
    return {'id':'learn/shading/energy-conservation','updated_on':'2026-09-07',
            'source_sha256':fingerprint(),
            'html':body}

def lighting_section():
    return r'''<section id="light-recipes"><h2>8. Point lights and distant lights</h2>
<p>A distant light sends parallel rays: moving a surface does not change the light's specified strength.
A point source spreads its power over larger areas as you move away. Double the distance and the same
power is distributed across four times the area. Radiant intensity describes the source's power per solid
angle; normal-incidence irradiance describes arriving power per perpendicular area. These are not
interchangeable meanings for a number called brightness.</p>
@@q=(s-p)\cdot(s-p),\quad L=\frac{s-p}{\sqrt{q}},\quad W=\frac{J}{q}@@s is the source position, p the receiving point, q squared distance, L the unit direction toward the light and J the RGB radiant intensity. W is the arriving delta-light weight before the surface cosine, distinct from irradiance I used earlier.@@
<p>The material still multiplies W by its BRDF and the positive normal-light cosine once. A zero distance
has no defined direction and infinite ideal falloff: our planned M1 query returns an invalid mask and zero
payload for that pair. This is not a physically dark lamp. Consumers must keep the mask. No hidden epsilon
turns a coincident lamp into a large finite number. Normals and explicit light directions are normalized
as part of these M1 recipes; this does not change the older renderer's separate arithmetic convention.</p>
<h3>Worked example</h3><p>For intensity 4, a point one unit away receives weight 4. At two units it receives
1; at half a unit it receives 16. A distant light configured to 4 remains 4. These are weights before the
surface cosine and the material, not final displayed pixel values. Keep coordinate units consistent.</p>
<h2 id="light-sh">9. Spherical harmonics: nine knobs per color</h2>
<p>An environment can illuminate a surface from every direction. Storing a full image of the surrounding
world is one representation. A low-order spherical-harmonic expansion instead stores weights of smooth
patterns over the sphere, much as a few musical harmonics approximate a complicated waveform. Band two
has nine coefficients for red, nine for green and nine for blue: 27 numbers per environment. Coefficients
may be negative because the basis patterns themselves have positive and negative regions.</p>
@@I(N)=\sum_{k=0}^{8} A_k c_k Y_k(N)@@I is cosine-convolved irradiance, not the dimensionless energy fraction E in the material budget. N is a unit surface normal, c incident-radiance coefficients and Y the real basis. The factors are pi for band zero, 2pi/3 for band one and pi/4 for band two. This is a Lambert convolution, not complete principled-material shading.@@
<p>Our first-order basis uses y, z, x in that order. Other renderers may use different signs, order or
pre-scaled coefficients. Copying 27 numbers between them without converting conventions can produce
plausible but wrong illumination. Batch B, sample point P and coefficient K are distinct axes: [B,P,3]
normals paired with [B,9,3] coefficients produce [B,P,3] irradiance. Shared coefficients explicitly use a
singleton batch or [9,3]; no hidden mixing of different scenes is allowed.</p>
<h3>Worked example</h3><p>For unit constant radiance, set the first coefficient to sqrt(4pi) in each channel
and the rest to zero. Irradiance becomes pi in every direction; multiplying by white Lambert 1/pi returns
one. A sharp directional source is different. Its band-two approximation has this response:</p>
@@K(t)=\frac{1}{4}+\frac{t}{2}+\frac{5(3t^2-1)}{32}@@t is the cosine between normal and source direction. K is the truncated response, not max(t,0).@@
<p>At t=1 it returns 17/16 rather than one. At t=-1 it returns 1/16 rather than zero. At t=-8/15 it is
-19/480: slight negative ringing. We preserve signed results rather than clamping them and silently changing
their derivatives. A fitted nonnegative environment needs an explicit constraint or different representation.
The demo colors negative areas magenta while retaining actual signed values in its graph.</p>
<details><summary>Try it yourself</summary><p>If the point-source intensity doubles while distance doubles,
does the received weight stay constant? Can nine coefficients exactly reproduce the hard light boundary?</p>
<h3>Answer</h3><p>The weight halves: twice the intensity divided by four times the squared-distance factor.
No: nine smooth basis functions approximate the hard boundary. Its overshoot and negative ringing are
properties to measure, not evidence that an exact directional-light formula should return negative light.</p></details>
<h3>The high-resolution lighting study</h3><p>Light reveals form adds a reciprocal Phong highlight to the development presentation. For unit normal N, source direction L and view V, reflect L as R=2(N dot L)N-L. Its lobe is max(R dot V,0) raised to exponent 72, scaled by 73/(2*pi). The unweighted positive-lobe integral is at most 2*pi/73; multiplying by the incident cosine cannot increase it. A 0.12 highlight budget plus 0.88 diffuse budget with albedo at most one therefore has a conservative unit energy bound. Reciprocity follows because R(L) dot V equals R(V) dot L. Numerical hemisphere integration and reciprocity tests exercise the same helper used for the images.</p><p>This is a deliberately bounded demonstration recipe, not the pending public principled material. The background, tone mapping and edge smoothing are presentation choices; no shadow solver or indirect illumination is claimed. The source bunny receives one offline display refinement, not a newly implemented public subdivision operation.</p>
<h3>gffx boundary and Follow the evidence</h3><p>These are executable contract recipes with interior gradient
checks, not a public gffx light implementation, native/GPU parity claim or completed renderer. Coincidence
and invalid-domain boundaries do not have ordinary derivatives. The full principled environment integral
cannot reuse Lambert's convolution factors without a separate validated approximation.</p><ul>
<li><a href="/testing/?page=shading/light-contract">Light equations, batch ownership, gradients and singularity checks</a></li>
<li><a href="/testing/?page=shading/lights-demo">Moving-light studio, distance curves and analytic spheres</a></li>
<li><a href="/testing/?page=camera/studio-demo">Camera studio: rendered color, normals and depth</a>. The room uses native gffx rasterization and perspective-correct interpolation, plus demo-only Lambert lighting. The yellow cross marks the point source as a display annotation. The image applies x/(1+x) and gamma 1/2.2 for display; it is not an energy measurement. No shadows, indirect bounce, public material API or Cornell-reference equivalence is implied.</li>
<li>Executable recipes: tools/light_contract_reference.py; fixtures: tests/python/test_light_contract.py.</li>
<li><a href="https://pbr-book.org/4ed/Light_Sources/Point_Lights">PBRT: point sources, radiant intensity and delta distributions</a></li>
<li><a href="https://graphics.stanford.edu/papers/envmap/envmap.pdf">Ramamoorthi and Hanrahan: low-order irradiance environment maps</a></li>
</ul></section>'''


def lobe_section():
    return '''<section id="lobe-building-blocks"><h2>From light to surface response: five building blocks</h2>
<p>Prerequisites: dot products, unit directions, radiance versus irradiance, and the energy-budget discussion above.
A BRDF says how strongly a surface redirects light between two directions. It is not an image, a light,
or a shadow test. The renderer still multiplies the incident light weight and surface cosine once.</p>
<h3>1. Matte reflection and the half-vector</h3>
<p>Lambert has a direction-independent BRDF: albedo divided by pi. Power into a solid angle
still carries the surface cosine; it is not uniform power per direction.
For albedo 0.5 its BRDF is 0.5/pi, and a constant unit-radiance environment returns 0.5.
For highlights, V points toward the camera and L toward the light. The half-vector H is the normalized
sum V+L; it points between them. At a normal-facing camera and a light 60 degrees away, H is only
30 degrees from the normal. Blinn-Phong uses N dot H, unlike reflected-vector Phong.</p>'''+equation(
    r'B(c,n)=\frac{n+8}{8\pi}c^n',
    'Classic highlight: c is the normal-half-vector cosine; n is a nonnegative real exponent.')+'''
<p>A larger exponent makes a narrower, taller highlight. This conventional approximate scale is not
an energy guarantee. With exponent 2 and normal view, c squared=(1+z)/2 where z=N dot L.
Integrating over light directions gives 25/24, about 4.17 percent too much. Our test keeps that failure
visible. Do not substitute this classic approximation for the conserving principled model.</p>
<h3>2. Distribution, visibility and Fresnel are different factors</h3>
<p>GGX D describes how microscopic surface normals are distributed. Its projected-area integral is one;
this does not mean its raw values sum to one or that it is a complete BRDF. Perceptual roughness r maps
to alpha=r squared, so the numerator uses r to the fourth power. At r=1, D is 1/pi.</p>
<p>Height-correlated Smith visibility describes geometric masking and shadowing of those microscopic
facets. It is not the scene's shadow map. Our visibility factor already includes division by
4 times the view cosine times the light cosine; never divide twice. At r=1 it reduces to
1/(2*(nV+nL)) in the positive hemisphere. Schlick Fresnel describes angular reflectance:</p>'''+equation(
    r'F(c)=F_0+(1-F_0)(1-c)^5',
    'Schlick reflectance: c is V dot H and F0 is normal-incidence RGB reflectance.')+'''
<p>For F0=0.04 and c=0.5, F=0.07. At c=1 it is F0; at c=0 it is one. Multiply D,
Smith visibility and F to obtain the single-scattering GGX specular BRDF. Under a unit environment,
roughness one, normal view and F=1, its exact outgoing value is 1-log(2), approximately 0.30685.
The missing energy is not a numerical bug: this model omits multiple scattering. The separately
specified principled coupling assigns a residual matte budget; these elementary factors do not.</p>
<h3>3. Boundaries and gradients are part of the operation</h3>
<p>Functions accept finite float32/64 tensors with explicit point and batch axes. They do not turn
negative cosines into positive ones, normalize input vectors or broadcast a new scene silently.
The material layer will own direction preparation and two-sided decisions. Roughness zero is a
delta limit, not a tiny hidden epsilon. A positive value that loses a needed divisor in the chosen
dtype is an explicit error. Empty and strided tensors are still valid layouts.</p>
<p>Smith returns zero at either horizon with zero branch derivatives. Blinn uses zero derivatives
at zero cosine, including its special exponent-zero corner. These conventions avoid NaNs but do
not claim smooth physics at a discontinuity. Lambert and Schlick retain their algebraic endpoint
derivatives. Interior finite-difference tests are different evidence from endpoint policy tests.</p>
<details><summary>Exercise: compute Schlick and distinguish two normalization tests</summary>
<p>For F0=0.2 and c=0.5, F=0.225. A GGX projected-NDF integral of one tests microfacet area;
a hemisphere BRDF integral at most one tests energy. Passing the first does not establish the second.
At exponent 2 the classic approximate Blinn response gives 25/24, so it fails a strict energy ceiling.</p></details>
<h3>Implementation boundary and evidence</h3><p>These are executable Phase-0 recipes, not yet public
gffx.shade APIs, a completed principled material, CUDA conformance or a compiler/export guarantee.
The four-sphere plate uses analytic normals and a declared display transform, not mesh rendering.</p>
<ul><li><a href="/testing/?page=shading/lobe-contract">Elementary formulas, boundaries and gradients</a></li>
<li><a href="/testing/?page=shading/lobes-demo">How a surface catches light: four spheres and angle curves</a></li>
<li><a href="https://google.github.io/filament/main/filament.html">Filament: microfacet factors and conventions</a></li>
<li><a href="https://seblagarde.wordpress.com/2011/08/17/hello-world/">Lagarde: normalization choices and their limitations</a></li></ul>
<p>Source: tools/shade_contract_reference.py; independent fixtures: tests/python/test_shade_contract.py.</p></section>'''


def budget_section():
    return '''<section><h2>Differentiating the energy accountant</h2>
<p>A material optimizer must know more than how bright a highlight is. When roughness changes,
the shiny part reflects a different fraction of the incoming light, so the matte part's remaining
budget changes too. Computing a budget once and detaching it from autodiff gives the optimizer
an incomplete derivative even if the displayed forward image looks plausible.</p>
<p>Our new reference uses fixed light directions and integration weights independent of roughness,
Fresnel reflectance and view cosine. Autodiff passes through every term of that weighted sum.
Only constant quadrature nodes are cached. This avoids differentiating a moving sample-rejection
boundary, but it can need many samples to resolve a narrow highlight. It is a correctness tool,
not the shipping fast path or a real-time claim.</p>'''+equation(
    r'E(c)=1-c\log\left(\frac{1+c}{c}\right)',
    'Independent control for roughness one and Fresnel fixed to one; c is the positive view cosine.')+equation(
    r'\frac{dE}{dc}=-\log\left(\frac{1+c}{c}\right)+\frac{1}{1+c}',
    'An exact derivative can detect mistakes that differentiating a numerical approximation cannot.')+'''
<p>At c=0.5 the reflected fraction is 1-0.5 log(3), about 0.450694. Its derivative is
-log(3)+2/3, about -0.431946. The cosine-weighted average across views is
4(1-log(2))/3, about 0.409137. These are single-scattering specular values, not the
complete principled material. White matte fills the residual in the exact coupled model.</p>
<h3>Three different questions need three different tests</h3>
<ol><li>Does the computation differentiate its own formula? Interior gradcheck tests this.</li>
<li>Does that formula approximate the intended integral? Independent visible-normal quadrature
checks held-out values; refined direct-grid finite differences check derivatives.</li>
<li>Does the complete material conserve energy? The separate independent furnace must still pass.
A graph of E plus 1-E being one proves only arithmetic, not the accuracy of E.</li></ol>
<p>The first reference run failed two refinement checks; increasing quadrature precision fixed
them without changing the limits. No finite test grid proves every roughness or grazing limit.
Current derivative evidence uses scalar float64 CPU tensors and positive interior view cosines;
world-normal derivatives, public RGB/batch layouts and fast approximations remain separate work.</p>
<details><summary>Exercise: what does freezing the budget lose?</summary><p>If white matte has
budget 1-E, its derivative with respect to roughness must be the negative of the shiny budget's
derivative. Freezing E makes that response zero. The total may still look correct at the original
parameter value while giving the optimizer the wrong direction to move.</p></details>
<ul><li><a href="/testing/?page=shading/budget-contract">Independent budget values and gradients</a></li>
<li><a href="/testing/?page=shading/budget-demo">Where the light budget goes: finite curves</a></li></ul>
<p>Source: tools/material_budget_reference.py; tests/python/test_material_budget.py.
This section does not announce public material APIs or a completed shading phase.</p></section>'''


def material_section():
    return '''<section><h2>Material values meet the surrounding light</h2>
<p>Colour, roughness and reflectance are fields on a surface. A field may be constant, interpolated
from triangle corners, or sampled from a UV texture. The material receives the resulting values
at each surface point, not an image filename or an instruction to guess texture channels.</p>
<p>Albedo has three linear RGB channels. A uniform specular or F0 scalar explicitly means grey;
a per-point grey field keeps a trailing one-channel axis so three points cannot be confused with
three colours. Classic uses a highlight strength and an explicit exponent. Principled uses
roughness and normal-incidence reflectance F0; these are not interchangeable meanings of specular.</p>
<p>Sample perceptual roughness first, then square it to obtain alpha for GGX. Squaring before
interpolation gives a different material. Two equal weights on roughness 0.2 and 0.8 give r=0.5,
then alpha=0.25; averaging the two pre-squared values would give 0.34 instead.</p>
<h3>Textures are already supported; material wiring must respect them</h3>
<p>The existing sampler reads level 0 unless LOD or derivatives are supplied. Keep UV seams as
face-corner attributes, interpolate UVs, then sample. Colour uses three channels; roughness and
grey reflectance use separate one-channel maps. The current native sampler supports at most four
channels per call. Our first five-channel fixture was rejected; splitting the maps keeps that
implementation limit explicit. Invalid background fragments stay outside material evaluation:
their zero-filled roughness is not a valid rough surface.</p>
<h3>Lighting integration needs both a direction and its share of the sky</h3>
<p>A quadrature weight represents a small solid angle, measured in steradians. It is not a PDF,
point-light intensity, or a set of numbers that should be normalized to sum to one. Evaluate the
incoming radiance and complete material response at each direction, then apply the surface cosine
and solid angle exactly once:</p>'''+equation(
    r'L_o=\sum_q f_r(V,L_q)L_i(L_q)\max(0,N\cdot L_q)w_q',
    'fr is the complete BRDF; Li is incident radiance; w is solid angle, not a probability density.')+'''
<p>N, V and L are unit directions in one coordinate system. The reference normalizes nonzero
vectors explicitly, keeps signed SH reconstruction, and never adds shadows or display gamma.
The cosine derivative is declared zero exactly at the horizon. A separate material evaluator
must also handle its view-side boundary; this integration helper cannot infer it from samples.</p>
<p>Our nine SH numbers per colour describe incident radiance. Lambert's convenient band factors
already integrate one particular cosine-shaped response. Applying them to the shiny-plus-coupled-matte
material changes the model. A nonconstant sky, Li=1+0.5*z, exposes that shortcut in an analytic
roughness-one control. Incoming radiance reaches 1.5, so an outgoing value above one here is not
by itself an energy violation. The unit-white furnace remains a different test.</p>
<details><summary>Exercise: why not normalize quadrature weights?</summary>
<p>For a unit Lambert surface under unit radiance, the hemisphere integral of the cosine is pi.
The BRDF is 1/pi, so reflected radiance is one. Rescaling all weights scales the answer; dividing
hemisphere solid angles by their sum 2*pi would produce 1/(2*pi), not one.</p></details>
<p>Current evidence covers field preparation, native interpolation/texture-to-field gradients,
and environment summation. It does not close general material evaluation, fast budget selection,
two-sided rules, scene acceptance, CUDA or compilation/export.</p>
<ul><li><a href="/testing/?page=shading/material-inputs">Material inputs and environment tests</a></li>
<li><a href="/testing/?page=shading/environment-demo">Full material versus the Lambert shortcut</a></li></ul>
<p>Sources: tools/material_contract_reference.py and tests/python/test_material_inputs.py;
formulas specialize the energy coupling derived above.</p></section>'''


def table_section():
    return '''<section><h2>A small table can still have gradients</h2>
<p>Imagine measuring the shiny energy budget at a grid of roughness and viewing-angle settings,
then answering nearby queries by interpolation. The saved numbers are constants, but the query
coordinates are not. Moving a roughness value changes its interpolation weights, so autodiff
can still see how the answer changes. That differs from detaching the answer after looking it up.</p>'''+equation(
    r'E(c,r,F_0)=E_0(c,r)+F_0\left(E_1(c,r)-E_0(c,r)\right)',
    'Schlick makes the budget affine in F0: store the F0=0 and F0=1 endpoints.')+'''
<p>For example, endpoint budgets 0.1 and 0.8 give 0.24 at F0=0.2, with derivative 0.7.
The F0 interpolation is exact for this model; interpolation across roughness and view cosine
is an approximation. The current trial uses a local three-by-three quadratic stencil, so values
are continuous but slopes can change where the stencil changes. The old bilinear option remains
selectable. A nice-looking value curve does not establish an accurate derivative.</p>
<p>The average is integrated exactly from each quadratic cosine segment times its cosine weight,
using two Gauss nodes per interval. This keeps
the two budget calculations consistent. It still does not prove the table agrees with the original
GGX integral. We separately test held-out values and derivatives, then repeat the independent
108-case furnace without changing its limits.</p>
<h3>Test the measuring stick as well as the shortcut</h3>
<p>An expensive reference calculation is still a numerical approximation. Before blaming a table,
increase the reference resolution and check whether its value and derivatives settle down.
If they do not, the result is an unresolved reference, not evidence that the shortcut is accurate
or inaccurate. Our expanded checks separate this guard from the table's own accuracy limits.</p>
<p>In the glossy, grazing-angle examples, uniformly spaced azimuth samples miss a narrow
reflection peak. The integrand depends on azimuth through its cosine, so we can integrate
half the circle and double the measure. A fixed Gauss grid concentrates samples near the
endpoints, resolving that peak with fewer samples. The coordinates stay independent of the
material parameters, so ordinary autodiff still applies. We keep the old rule as a negative
control and compare the refined values with a separate visible-normal integration.</p>
<p><a href="/testing/?page=shading/reference-demo">See the two angular grids converge differently</a>.
An improved reference can expose a real table error that was previously hidden by an unreliable
comparison; it does not automatically make the table pass.</p>
<p>The next experiment doubled the angle samples and shared geometric integration work between
the two Fresnel endpoints. It fixed the original error, but more grazing queries exposed three
failures. Redistributing roughness knots then made sixteen checks fail, including the furnace
refinement check. Neither failure was solved by relaxing a tolerance. Local quadratic interpolation
on uniform roughness knots passes the expanded screening with the same stored sample count.
It reads nine endpoint pairs per query instead of four. Quadratic interpolation is not generally
shape-preserving: a finite dense energy-range screen supplements, but does not replace, the furnace
and reference checks. These results do not prove accuracy everywhere between the test points.</p>
<p>Internal gradcheck asks whether autodiff differentiates the implemented approximation correctly.
Reference comparison asks whether that approximation has the right physical slope. These are
different questions. At a table knot, compare the declared one-sided slope instead of pretending
there is a unique two-sided derivative. Finally, test the complete shiny-plus-matte response:
normalizing directions and coupling both energy budgets must keep their chain-rule contributions.
A deliberately detached budget is a useful wrong-answer control.</p>
<h3>Scope is part of correctness</h3><p>This candidate covers roughness 0.2 through 1,
not zero roughness or every sharper material. Queries outside it fail rather than clamping.
Gffx's physical domain has not been narrowed: a strategy for sharper materials must be resolved
before adopting a public implementation. The table is currently a CPU development object,
not an installed package resource, GPU result or compile/export guarantee.</p>
<p>The default candidate stores 138544 bytes of float64 knot, endpoint and average tensors
(33 roughness by 257 cosine knots, plus axes and means),
excluding Python object overhead and query temporaries. Expensive generation happens explicitly,
once per constructed object; it must not run in a frame loop. Local timing records separate
construction, batched queries and backward passes. They are not rendering frames per second.</p>
<details><summary>Exercise: can constant table data give a nonzero gradient?</summary>
<p>Yes. Interpolating values 2 and 6 over coordinates 0 and 1 gives 2+4*x. The data are constants,
but its derivative with respect to x is 4. Detaching the final interpolated result would instead
lose that derivative. At a knot, a piecewise-linear table needs a declared one-sided slope convention.</p></details>
<p>One low-precision generation produced 1.0000000000000013 at a unit-Fresnel horizon.
The out-of-range guard rejected that rounding result; it was not clamped into acceptance.
The historical rejection is retained while higher-precision generation and deterministic
injected-error tests check the intended guard. Neither incident proves physical overreflection.</p>
<ul><li><a href="/testing/?page=shading/table-contract">Candidate accuracy, gradients and energy tests</a></li>
<li><a href="/testing/?page=shading/table-demo">Compare values, slopes and local query timings</a></li></ul>
<p>Source: tools/material_budget_table.py and tests/python/test_material_budget_table.py.
This candidate does not close material implementation or Phase 0.</p></section>'''


def sharp_section():
    return '''<section><h2>Polished surfaces: move the coordinates, not the material</h2>
<p>A very smooth surface reflects light into a tiny angular region. A regular grid can step
over that region entirely. Making the surface rougher would make integration easier, but would
change the material we promised to evaluate. Instead, choose coordinates that follow its reflection.</p>
<p>Describe a microfacet's half-vector by its slope t and azimuth. Let c be the view cosine,
and b the view's tangential component along that azimuth. A reflected ray is above the surface when:</p>'''+equation(
        r'c(1-t^2)+2bt>0',
        'The positive root bounds the half-vector slopes that reflect into the upper hemisphere.')+'''
<p>Scale the slope by alpha=r squared, then use an angle beta with t=alpha*tan(beta).
This concentrates integration around the narrow GGX peak. The largest allowed beta depends on
the view and roughness. Keep that changing boundary in the autodiff graph: freezing it would
discard part of the derivative. Constant unit-interval Gauss nodes are mapped through that boundary.</p>
<p>A second, independent visible-normal integration initially failed to converge in three comparison
cases. Redistributing its radius-squared samples toward the unit-disk edge resolves those cases
without adding samples or relaxing the limits. The old rule remains selectable and its failure
is a regression control. Agreement matters only when both measuring methods are resolved.</p>
<p>The sharper reference passes sixty sampled value/gradient comparisons at roughness 0.01 through
0.199 and a separate 108-case shiny-plus-matte energy study. Its average is integrated over view
directions, including roughness and Fresnel gradients. Exact roughness-one controls and a mirror-limit
example check different parts of the calculation. These are finite tests, not a proof for every
positive roughness or a completed public material.</p>
<p>The old fast table still rejects roughness below 0.2. A reliable offline reference is the
measuring stick for its eventual extension, not an excuse to run expensive integration inside
every pixel. Near-perfect reflection also leaves a tiny matte budget: subtracting almost equal
numbers and dividing by that small remainder needs explicit conditioning checks.</p>
<details><summary>Why does the polished-sphere picture not prove energy conservation?</summary>
<p>The picture uses 512 deterministic environment samples per pixel, analytic spheres and
an illustrative display transform. It can show reflection shape but can still have integration
error. The energy tests separately integrate independent specular reflection plus the coupled
matte response under a white environment and enforce the original numerical limits.</p></details>
<ul><li><a href="/testing/?page=shading/sharp-demo">See four polished surfaces and their budgets</a></li>
<li><a href="/testing/?page=shading/sharp-contract">Inspect the sharp reference and energy tests</a></li></ul>
</section>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for identifier,result in articles().items():
        output=ROOT/'static/learn'/asset_name(identifier)
        if args.check:
            if json.loads(output.read_text(encoding='utf-8'))!=result:raise SystemExit('Generated article is stale: '+identifier)
        else:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Textbook is reproducible' if args.check else 'Built 15 textbook pages')

if __name__=='__main__':main()
