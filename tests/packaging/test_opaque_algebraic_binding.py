"""Original producer exposes exact role inputs without selecting new acceptance."""
import pytest
@pytest.mark.parametrize('material_only',[False,True])
def test_original_algebraic_helpers_preserve_prepared_roles(monkeypatch,material_only):
    import os,json,hashlib
    from pathlib import Path
    from fractions import Fraction as Q
    import pytest
    from tools import opaque_full_material_reference as m
    from tools.opaque_shading_intervals import I,Jet
    source=Path(os.environ['GFFX_FULL_INPUT']);assert hashlib.sha256(source.read_bytes()).hexdigest()=='a15bea40cb5cdead7994d828b57f5dcfac9f44fab0292a06e7d0a7b08719f73a'
    class Stop(Exception):pass
    def capture(field,**kw):
        c=kw['correction_integrator'];assert callable(c.role_inputs) and callable(c.implicit_inputs)
        box=(I('.75','.78125'),I('.75',1));kw['prepare'](0,0,box)
        for order in (0,1,3,10):
            u=Jet.variable(box[0],order=order) if order else Jet(box[0]);v=Jet(box[1],order=order);old=c.inputs(u,v)
            for role in (0,1):
                new=c.role_inputs(role,u,v)
                assert all((old[role][k].lo,old[role][k].hi)==(new[k].lo,new[k].hi) for k in set(old[role].c)|set(new.c))
        raise Stop
    monkeypatch.setattr(m,'integrate_square',capture)
    with pytest.raises(Stop):m.full_pixel(json.loads(source.read_bytes()),selected_piece=8,material_only_inputs=material_only)
