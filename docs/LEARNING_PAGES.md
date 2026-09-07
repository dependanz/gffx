# Learning pages and dependency-free mathematics

The repository testing application has a **Learn** group and a mini-textbook at
`/testing/?page=learn/index`. Its fourteen chapters cover vectors, transforms,
triangle geometry, topology, barycentrics, cameras, rasterization, textures,
proximity, surface sampling, autodiff, numerical verification, energy conservation,
and motion. Prerequisites plus previous/next links provide a reading path. Each
chapter includes worked examples, an exercise with an answer, implementation
boundaries and links to relevant tests or finite demonstrations. The original
energy article retains its stable `learn/shading/energy-conservation` identity.
It now also explains point and distant lights, inverse-square units, the nine
SH coefficients, signed approximation error, batch ownership and zero-distance
handling. See `LIGHTING_CONTRACT_RECIPES.md` for the checked examples and finite demo.

These are authored articles, not passing tests, execution artifacts or implemented
shading APIs. Article measurements are dated; linked test pages may contain newer
results. Articles exist without a test execution, appear in the creation-date list,
and do not create runs or persist updates. Ordinary dashboard retention is unchanged.

## No new external dependency

`tools/testing_dashboard/mathml.py` converts a deliberately limited LaTeX-math
syntax to native MathML using only Python's standard library. The browser lays out
that MathML. There is no KaTeX, MathJax, TeX installation, CDN, downloaded font,
JavaScript package or network rendering service. This is repository developer
tooling, not a new public `gffx` API or a dependency of `gffx.dashboard`.

This does not mean gffx's optional PyTorch/CUDA adapters or its test/build tools
have no prerequisites. Their existing dependency boundaries are unchanged.

## Supported math subset

- Numbers, identifiers, arithmetic/comparison punctuation and selected Greek letters.
- Grouping, subscripts and superscripts: `x_i`, `x^{12}`, `x_i^2`.
- `\frac{a}{b}`, `\sqrt{x}`, `\sum`, `\prod`, `\int` and limits via scripts.
- Selected named operators such as `\cdot`, `\times`, `\leq`, `\partial`, `\nabla`;
  functions such as `\sin`, `\cos`, `\log` and `\exp`.
- `\mathbf{N}`, `\mathrm{R}`, `\text{ordinary words}`, `\operatorname{foo}`;
  `\vec{x}`, `\hat{x}`, `\bar{x}`.
- Matched `\left`/`\right` delimiters, including an invisible `.` side.
- `matrix`, `pmatrix`, `bmatrix`, `aligned`, `cases`: `&` separates cells and
  `\\` separates rows. Rows must have equal cell counts.
- A small set of explicit spacing commands.

The symbol tables in `mathml.py` are the exact list. Always brace multi-token
scripts. Numeric literals are one token here (so unbraced `x^12` differs from TeX).
Text groups cannot nest. This is **not full LaTeX**: no macros, packages, documents,
file access, URLs, HTML injection or arbitrary execution. Unknown commands and
malformed syntax raise `ValueError`, with source offsets for parser errors; they
are not silently dropped. Limits are 8,192 source characters, 4,096 tokens and
48 nested math rows or atoms (including unbraced accents). Bold identifiers use Unicode mathematical characters rather
than legacy MathML styling. Glyph availability and accessibility vary by browser
and operating system; the readable LaTeX source is retained alongside each formula.

## Authoring and verification

Chapters and prerequisite/evidence metadata live in `tools/testing_dashboard/textbook.py`.
The original energy lesson and shared generator live in `build_lessons.py`.
Equation placeholders have the form `@@latex@@plain-language caption@@`.
The generator escapes the caption and original LaTeX and creates structural
MathML. Regenerate the checked-in JSON after editing the article or converter:

```powershell
python -m tools.testing_dashboard.build_lessons
python -m tools.testing_dashboard.build_lessons --check
python -m pytest tests/python/test_textbook.py tests/python/test_learning_pages.py tests/python/test_testing_dashboard.py -q
```

Commit source and generated JSON together only through the project's approved
publication process. The JSON includes the article date and SHA-256 of all three
authoring sources; the freshness test rejects drift. Article HTML is limited to
100,000 characters at the frontend and passes a strict element/attribute allowlist.
Only authored local articles may supply this HTML, never dashboard producer data.
No active scripts, event attributes, arbitrary external resources or unsafe links
are accepted. Interactive behavior is implemented by the testing application.
Section links use a portable fragment ID (letters, digits, underscore and hyphen,
starting with a letter) and must refer to a target inside that same article. They
are not restricted to an energy-chapter prefix. The reader collects all targets
before checking links so the contents can link forward to later sections. Tests
apply the actual frontend link policy to every generated lesson as well as unsafe
and missing-target fixtures; source-generation freshness alone does not check this.

For another lesson, extend `CHAPTERS` with a unique stable `learn/...` ID, earlier
prerequisites, evidence links, source pointer and authored body. The generator,
fixed asset-name registry, catalog and reading-order navigation derive from it.
Assign the actual immutable creation date when extending the catalog in the future;
the initial textbook batch shares 2026-09-07. Add/update the related feature's
`learning_article` link, parameterized freshness test, curriculum-size expectation,
independent worked-example arithmetic and explanation in `explanations.py` as needed.
Check all asset basenames stay unique. Do not accept user-controlled file paths.

For every new graphics capability, write or update its chapter as part of completion:
intuition, prerequisites, symbols and units, derivation, checked numerical example,
exercise/answer, singularities, gradient limits, exact implementation boundary and
relevant tests/demos. Keep future material marked as planned until authored and
reviewed. The book explains the project; it is not a second authoritative plan.

Article refreshes preserve open derivations and calculator state while the source
hash is unchanged. Long equations scroll inside their own cards, not the whole
page. Controls have touch-sized targets and honor the existing light/dark theme.
Browser layout checks do not replace physical iPhone/iPad acceptance, nor does a
rendered equation prove the mathematics or a graphics implementation correct.
