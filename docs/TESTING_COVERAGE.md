# Repository coverage in the testing dashboard

The testing application is repository tooling (`tools.testing_dashboard`), not a new
runtime dependency of `gffx.dashboard`. Its coverage reader can import Python JSON
and standard LCOV reports from other repositories, but its page catalog and native
fixture collector remain gffx-specific. This is not yet a standalone any-repository product.

## What the numbers mean

Choose Python, Native C/C++, JavaScript/TypeScript, CUDA, shaders, or another instrumented
layer on the coverage page. Each percentage uses **that report's executable lines**.
Branch and function coverage appear when supplied. Python uses coverage.py's statement
summary, not the length of its raw executed-line list. There is no combined percentage:
different tools and languages do not have interchangeable denominators.

The Git inventory includes tracked and nonignored source files, including tests and
headers. A listed file absent from a report is **not represented**, not 0% covered.
CUDA and shader execution are not measured merely because host tests use a GPU.
Coverage tells you what executed, not whether the result is correct, how fast it is,
or which individual test exercised a line. Reports belong to an explicit execution;
missing layers never silently borrow an older execution's results.

## Python: package plus repository tools

From the repository, using the intended installed wheel and optional `coverage` developer
tool (the existing `testing-dashboard` dependency group):

```powershell
python -m tools.testing_dashboard run --address http://127.0.0.1:8791 --run my-fresh-check --coverage -- tests/python tests/pytorch tests/packaging -q
```

The default source selection is `gffx,tools`. Override it with repeatable
`--coverage-source my_package --coverage-source tools`. Reports retain their SHA-256;
installed gffx paths are attributed to `src/gffx`. Keep the raw report and the exact
source/wheel provenance. The CLI does not rebuild the wheel for you.

## JavaScript: Node's built-in coverage

This profile uses the installed shared viewer and repository tester. Set
`GFFX_DASHBOARD_STATIC` to the selected installed wheel's `gffx/dashboard/static` directory.
Run from the repository with an existing Node runtime:

```powershell
node --experimental-default-type=module --test --experimental-test-coverage --test-coverage-include='**/gffx/dashboard/static/*.js' --test-coverage-include='**/tools/testing_dashboard/static/*.js' --test-reporter=spec --test-reporter-destination=stdout --test-reporter=lcov --test-reporter-destination=build/javascript.lcov tests/dashboard/touch.test.mjs tests/dashboard/testing.test.mjs
python -m tools.testing_dashboard coverage --address http://127.0.0.1:8791 --run my-fresh-check --format lcov --layer javascript --inventory-root . --path-map build/MY-INSTALL/gffx src/gffx build/javascript.lcov
```

Create the output directory first and replace `MY-INSTALL` with the actual report prefix.
The checked Node 22 profile measures loaded file modules. It does not measure browser
WebGPU execution; tester `app.js` tests currently import a rewritten data URL, so that
file is explicitly absent from the LCOV source-file report even though tests exercise it.
Inspect the inventory gaps rather than treating this as whole-browser coverage.

## Native C: separate GCC fixture build

On an existing Linux/WSL system with Python, GCC and gcov (no installation is performed):

```text
python3 tools/testing_dashboard/native_coverage.py --root /path/to/gffx --output /path/to/gffx/build/native-coverage-FRESH
```

The output must be new. The profile instruments the C core at `-O0 --coverage`, executes
the geometry, PLY and cross-cutting C fixtures, and converts gcov JSON to `native.lcov`.
`commands.json` retains compiler/tool versions, every command and its result. This is
not the production CMake/release build, Windows coverage, C++ adapter coverage, or CUDA
device coverage. Keep this distinction when comparing reports.

```powershell
python -m tools.testing_dashboard coverage --address http://127.0.0.1:8791 --run my-fresh-check --format lcov --layer native --inventory-root . build/native-coverage-FRESH/native.lcov
```

## Additional instrumenters and safety

An external instrumenter can supply LCOV for any declared layer, including CUDA or
shaders only if it actually measures that code. The importer does not install tools,
execute report contents, compile arbitrary repositories, or infer backend correctness.
It bounds input to 8 MiB and 10,000 file records, validates counts, and rejects malformed
records. Imports replace only the selected layer. Use an execution with matching source
provenance; the importer cannot establish that an externally supplied report is genuine.
Prefix mappings are explicit and the raw report hash remains available.

For future coverage expansion, prioritize the unrepresented C++ adapter, browser-only
JavaScript shell, CUDA kernels, and WGSL shader. Their current state is **not measured**,
not a passing coverage gate. Production OS/compiler lanes remain separate CI evidence.
