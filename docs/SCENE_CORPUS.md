# Scene M1 source fixture entry

ID: `scene_hierarchy_m1`. Classification: first-party, source-controlled development
fixture, MIT. Source commit: not yet published; ancestor fc110903 is not claimed
to contain this fixture. This entry does not add an admitted rendered reference
to the external corpus registry or claim Cycles equivalence.

Generator: `tools/testing_dashboard/scene_demo.py`, SHA256
`6e40f568c30b8ca3182d761c40c5c0853bb085691e56756c769f4a875b2e3472`.
Invocation: `payload()`, CPU float64, one thread. Three cubes,24 vertices,
36 triangles,12 parent-transform frames. A192x192 native rasterized view and
an8x8 smooth triangle translation fit (eight fresh loss measurements) are finite
demonstrations. Stored playback frames are authored transforms, not simulation.
No external mesh, subject, FLAME model or downloaded asset.

Independent acceptance fixtures are in `tests/python/test_scene.py` (SHA256
`31e860a9575955e927b33e0249938648a4d28480eeb3081ba5ba9b9c5ac97d96`)
and `tests/python/test_scene_integration.py` (SHA256
`455f13dfa189a073f1aac2eee7d3ec240d913eb26507e7ea554f124893bce684`).
They use hand-computed hierarchy/packing, manual existing rasterization and
finite-difference derivatives; the demo image is not its own reference oracle.
Generated artifacts stay outside source Git; current measured outcomes and exact
installed artifacts are recorded in devbrain NUMBERS.md.
