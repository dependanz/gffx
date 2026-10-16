"""A bounded independent-target renderer fitting demonstration, not a consumer."""
import os,json
from pathlib import Path

def test_deferred_head_albedo_recovery_independent_target(record_property):
    from tools.deferred_recovery_demo import solve
    result=solve(Path(os.environ['GFFX_REFERENCE_CORPORA']),Path(os.environ['GFFX_DEFERRED_RECOVERY_OUTPUT']))
    assert result['steps']==20
    assert result['final_mse']<result['initial_mse']*.01
    assert result['max_parameter_error']<.03
    assert result['fixed_inputs_unchanged']
    record_property('renderer_recovery',json.dumps(result))
