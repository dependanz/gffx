"""Current-artifact replay of pinned historical optimizer evidence; no new solve."""
import ast,hashlib,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import torch


def test_saved_nine_parameter_solution_retains_current_image_acceptance(record_property):
    """A saved fit may transfer only through fresh evaluation against its independent target."""
    import test_image_full_recovery as recipe
    root=Path(os.environ['GFFX_IMAGE_EVIDENCE_ROOT'])
    report=root/'image10-focused1.json';xml=report.with_suffix('.xml')
    expected=('3610b41925471a4ea364b79551de5ddf7f1d10bc4f2c83ce1d28860edba8b9d4','4bdfd21d4261cf4b1515881398d8a198c5107d4e8803722911188895f0c5800a')
    for p,h in zip((report,xml),expected):assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    saved=json.loads(report.read_bytes());assert saved['status']=='passed' and saved['pytest_exit_code']==0
    assert saved['supervisor']['exit_code']==0 and saved['supervisor']['failure'] is None
    assert hashlib.sha256(Path(recipe.__file__).read_bytes()).hexdigest()==saved['tests']['tests/python/test_image_full_recovery.py']
    cases=[c for c in ET.parse(xml).getroot().iter('testcase') if c.attrib['name']=='test_full_image_recipe_recovers_and_repeats_with_locked_inputs']
    assert len(cases)==1 and not any(cases[0].find(k) is not None for k in ('failure','error','skipped'))
    props={p.attrib['name']:p.attrib['value'] for p in cases[0].findall('properties/property')}
    parameters=ast.literal_eval(props['recovered_parameters'])
    assert len(parameters)==len(recipe.NAMES)==9 and int(props['closure_evaluations_each_repeat'])<=100
    fixed=recipe.sources();target=torch.tensor(recipe.scalar_reference(fixed,recipe.TARGET),dtype=torch.float64)
    before=recipe._fixed_identity((*fixed,target));evaluations=[]
    for repeat in range(2):
        with torch.no_grad():
            initial=recipe.evaluate(fixed,target.new_tensor(recipe.INITIAL))
            final=recipe.evaluate(fixed,target.new_tensor(parameters))
        assert recipe._fixed_identity((*fixed,target))==before
        mse=[float((value-target).square().mean()) for value in (initial,final)]
        assert mse[0]>0 and mse[1]<=.01*mse[0]
        evaluations.append((recipe._bytes(initial),recipe._bytes(final),mse))
    assert evaluations[0]==evaluations[1]
    record_property('current_initial_mse',mse[0]);record_property('current_replayed_final_mse',mse[1])
    record_property('historical_solved_final_mse',props['final_fresh_mse'])
    record_property('historical_wheel_sha256',saved['wheel']['sha256'])
    record_property('historical_optimizer_closures_per_repeat',props['closure_evaluations_each_repeat'])
    record_property('recovery_report_sha256',expected[0]);record_property('recovery_xml_sha256',expected[1])
    record_property('scope','Two fresh evaluations of saved parameters; optimizer repeatability remains the pinned historical evidence. Independent target, original 1%-MSE acceptance and locked inputs retained.')
