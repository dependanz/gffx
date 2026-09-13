# Textured environment image continuation

The selected-tile checks covered 264 of 6,536 surface pixels in each of twelve
fixed material/light/albedo profiles. `tools.albedo_images` now owns the remaining
588 tiles: 98 create-only batches of six, 128 pixels per tile. Completed selected
tiles are reused, never relabelled as new work.

`tools.albedo_image_sampling` accepts independently converged references without
requiring previous public results. It calls real public material and lighting
operations. Separate regression tests compare against the 72 saved observations.
No prior public output is invented for a new pixel.

The measured route uses up to eight points for classic uniform/affine and
principled uniform; principled affine stays at one point because batching measured
slower. Principled uniform's small timing difference is not a performance promise.
Every call stays within 1,024 point/light pairs and the existing runtime admission.

Each worker is serial, one-thread CPU float64, with 120-second cooperative and
180-second watchdog limits and 1 GiB sampled whole-process host memory. Each batch
has 900 seconds with a 180-second launch reserve. A failure stops the batch and is
preserved for diagnosis, not automatically retried. References, source identities,
the original installed wheel and corpus remain pinned in every report.

The dashboard is a finite replay, not a running renderer. Reference is on the left,
public output on the right, gray uncomputed. Display uses half exposure then sRGB;
acceptance checks use unscaled linear RGB. A progress snapshot does not qualify
full images, image-loss derivatives, arbitrary HDR lighting or M1 completion.

`tools.albedo_image_assembly` closes one complete profile only after strictly
replaying all 52 unique tiles. It saves CPU float64 reference/public arrays with
exact zero background and recomputed image metrics. Replay reconstructs both
arrays byte-for-byte and checks report, tile, source, wheel and corpus identities.
The assembly fixture tests include missing/duplicate/reordered pixels and a
deliberately changed saved byte. One complete profile is not all twelve profiles.
