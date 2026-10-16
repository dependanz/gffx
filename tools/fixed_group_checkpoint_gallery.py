"""Display verified saved fixed-group outputs without running another renderer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    audit_path = args.root / 'verification2.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'passed'
    assert digest(args.root / 'jobs.json') == audit['jobs_sha256']
    jobs = json.loads((args.root / 'jobs.json').read_text())
    sources = [dict(path=str(audit_path), sha256=digest(audit_path))]

    def load(name):
        job = next(item for item in jobs if item['name'] == name)
        path = Path(job['path'])
        assert digest(path) == job['sha256']
        row = json.loads(path.read_text())
        assert row['status'] == 'passed' and row['stream']['complete']
        sources.append(dict(path=str(path), sha256=job['sha256']))
        values = {}
        for key, spec in row['arrays'].items():
            path = Path(spec['path'])
            assert path.stat().st_size == spec['bytes'] and digest(path) == spec['sha256']
            assert spec['dtype'] == '<f8'
            sources.append(dict(path=str(path), sha256=spec['sha256']))
            values[key] = np.frombuffer(path.read_bytes(), dtype='<f8').reshape(spec['shape'])[0]
        return row, values

    hard, h = load('cpu-512x512-0.0-1')
    soft, s = load('cpu-512x512-0.2-0')
    assert audit['comparisons'][2]['gradient_byte_identical'] is True
    image = Image.new('RGB', (1780, 1040), '#101a27')
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=23, fill='#b4c6d5', bold=False):
        font = ImageFont.truetype('C:/Windows/Fonts/' + ('seguisb.ttf' if bold else 'segoeui.ttf'), size)
        draw.text((x, y), value, font=font, fill=fill)

    text(54, 28, 'GFFX  /  M1 EXECUTION STUDY  /  19 SEPTEMBER 2026', 18, '#67ddd2', True)
    text(54, 68, 'Full-image gradients, bounded working memory', 43, '#f3f7fa', True)
    text(54, 131, 'Actual 512 × 512 outputs · 5,002 vertices · 10,000 triangles · float64 Principled shading', 24)
    for x, title, values in ((54, 'Hard coverage + albedo gradient', h), (608, 'Soft coverage · forward only', s)):
        text(x, 202, title, 26, '#f3f7fa', True)
        rgb = np.clip(values['colour'] + (1 - values['mask'][..., None]) * np.array([.007, .012, .02]), 0, 1)
        rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - .055)
        image.paste(Image.fromarray(np.rint(rgb * 255).astype('uint8')), (x, 254))
    text(54, 785, f"{hard['measured_seconds']:.3f} s · complete forward + backward", 25, '#67ddd2', True)
    text(54, 826, 'Colour, mask and gradient match prior bytes exactly.', 21)
    text(608, 785, f"{soft['measured_seconds']:.3f} s · complete forward", 25, '#67ddd2', True)
    text(608, 826, 'Original soft internal-edge limitations remain visible.', 21)
    text(1180, 211, '17 / 24', 67, '#67ddd2', True)
    text(1180, 296, 'required scale profiles qualified', 23)
    draw.line((1180, 355, 1724, 355), fill='#30475b', width=2)
    lines = [
        '128 points per internal group',
        '5 image rows per backward window',
        '103 windows, with reverse recomputation',
        '64 MiB window allowance unchanged',
        'CPU process cap: 1 GiB',
        '',
        'No pixels omitted; original camera retained.',
        'Four-point gradient accumulation is unchanged.',
        '',
        'Seven required profiles remain unqualified.',
        'Final showcase and M1 acceptance remain open.',
    ]
    for index, line in enumerate(lines):
        text(1180, 390 + index * 38, line, 22)
    draw.line((54, 904, 1724, 904), fill='#30475b', width=2)
    text(54, 927, 'Display only: linear RGB → sRGB on a dark background. Native pixels; no retouching or replacement imagery.', 22)
    text(54, 974, 'Engineering evidence: one evaluation per profile, constant RGB albedo and global RGB-mean loss; not arbitrary geometry gradients.', 21)
    path = args.output / 'plate.png'
    image.save(path, optimize=True)
    sha = digest(path)
    final = args.output / (sha + '.png')
    path.rename(final)
    caption = (
        'Actual CPU 512-square saved hard forward/backward and soft forward outputs. '
        'Hard colour, mask and constant-RGB-albedo gradient are byte-identical to the preserved '
        'qualified CUDA512 reference. Fixed 128-point grouping changes window scheduling only. '
        'The 64 MiB window allowance, 1 GiB CPU process cap and deadlines are unchanged. '
        'Display uses the same linear-to-sRGB conversion over a dark background for both images. '
        'Soft internal-edge limitations remain. This engineering checkpoint brings the required '
        'scale matrix to 17/24; final SIGGRAPH-quality demonstration acceptance remains open. '
        'New batch: 25 complete evaluations, 11 conservative admission refusals, with all evidence retained.'
    )
    item = dict(label='Bounded full-image gradients', caption=caption, path=str(final),
                asset=dict(url='/testing/media/' + sha + '.png', sha256=sha, bytes=final.stat().st_size))
    report = dict(status='passed', generator_sha256=digest(__file__), sources=sources, images=[item])
    rp = args.output / 'report.json'
    rp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest = dict(version=1, title='Bounded full-image gradients: 17 of 24 scale profiles',
                    page='render/scale-demo', scope=caption, images=[item],
                    provenance=dict(producer_report=str(rp), producer_report_sha256=digest(rp),
                                    wheel_sha256=hard['wheel']['sha256']))
    mp = args.output / 'manifest.json'
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(manifest=str(mp), sha256=digest(mp), image=str(final))))


if __name__ == '__main__':
    main()
