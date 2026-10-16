"""Both endpoints of the independently certified original pixel, installed wheel."""
def test_original_pixel_meets_independent_endpoints(record_property):
    import os,json,hashlib
    from pathlib import Path
    from fractions import Fraction as F
    from tests.python.test_opaque_full_original import test_complete_original_material_pixel
    test_complete_original_material_pixel(record_property)
    p=Path('D:\\devspace\\gffx\\qualification-20260913\\m1-continuation-20260913\\opaque120-20260924\\original-campaign1\\complete-certificate.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='5144bbd577543e3f9ee3ecd5fb992b13c29926a8eec7c1e18230e8d6e43b6249'
    ref=json.loads(p.read_bytes());actual=json.loads(Path(os.environ['GFFX_FULL_NATIVE']).read_bytes());rgb=actual['image']['colour'][0][0][0];jac=actual['albedo_jacobian'];checks=[]
    for label,value,pair in [(f'rgb{i}',rgb[i],ref['value'][i]) for i in range(3)]+[(f'albedo{i}',jac[i][i],ref['albedo_derivative']) for i in range(3)]:
        for endpoint in pair:
            error=abs(F.from_float(value)-F(endpoint));limit=F('1e-10')+F('1e-9')*abs(F(endpoint));checks.append(dict(channel=label,error=float(error),limit=float(limit),passed=error<=limit))
    assert all(jac[i][j]==0 for i in range(3) for j in range(3) if i!=j)
    record_property('original_accuracy',json.dumps(checks));assert all(c['passed'] for c in checks),checks
