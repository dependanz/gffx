# Larger CPU light batches

The material uses one current-graph curve per active point. Light directions gather
only the curve coefficients they need; RGB mixes the two Fresnel endpoints. This
permits more light samples without constructing another full curve for every sample.

The CPU float64 material limit is 1024 total `B*P*Q` point/light pairs, with at most
8 total points for forward-only calls or 4 for gradient-bearing calls. Counts are
checked before active masks. The caller workspace ceiling stays 64 MiB. Standalone
budget queries remain limited to 32 entries. No implicit batching, new dependency,
table change, formula change, persistent query cache or tolerance relaxation.

The conservative planning estimate is `3*budget.workspace(3*B*P, backward=mode)`
plus `8*B*P*Q*(192 if backward else 64)` bytes. Empty-light work uses zero query
entries. This is a planning estimate, not an OS reservation or measured allocator peak.

## Recorded qualification

The unchanged hardest reference workload contains 65,536 light directions in two
quadrature estimates. Explicit 1024-direction chunks reduce calls from 6554 to64.
The final-artifact sampling time is0.595788s versus37.8784s with the previous ten-light
chunks. Different reduction grouping causes tiny floating-point differences; the
maximum independent-reference error is2.1077862e-10, within the unchanged limits.
This is one strict point comparison, **not a rendered frame**.

Four full-capacity cases were measured with one warmup, three timed repetitions,
saved-autograd storage accounting, and twelve repeated calls:

| Points | Lights | Mode | Median | Planned bytes | Saved autograd bytes |
|---:|---:|---|---:|---:|---:|
| 1 | 1024 | forward | 6.786 ms | 6434048 | 0 |
| 8 | 128 | forward | 7.901 ms | 47802368 | 0 |
| 1 | 1024 | forward + backward | 15.955 ms | 15756288 | 1391426 |
| 4 | 256 | forward + backward | 19.998 ms | 58306560 | 2019464 |

The actual cost worker peaked at551813120 bytes; the environment worker at545218560
bytes, including Python/PyTorch. Both were supervised under the unchanged1GiB,
120s cooperative/180s watchdog limits. Saved storage includes shared inputs/constants;
it is not peak scratch. Finite repetition is not a leak-freedom proof.

Evidence lives in `build/light-batches-20260908/final-reports/{costs,environment}.json`, with exact
installed-wheel byte checks. Wheel SHA256:
`1c97f8dfa1f565078131a5e1192492680dbecd8d4ed8620c56d2ab4fa825c4c9`.
The first candidate's0.694s run and compatibility-error regression are preserved
beside final-reports; local timing variation is not an algorithmic change.
Independent reference/corpus bytes are unchanged. Full-image work must still declare
per-pixel convergence and total execution limits before it runs.
