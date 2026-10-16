"""Neighbour horizons must not alter another pixel's integration rule."""
import os
import pytest
import torch
from test_opaque_full_original import original

@pytest.mark.parametrize('x,width',[(23,7),(21,14)])
def test_horizon_window_equals_canonical_pixels(x,width):
    from gffx.torch import render
    s,c,_=original(os.environ.get('GFFX_KNOT_DEVICE','cpu'))
    with torch.no_grad():
        canonical=[render.opaque_window(s,c,(64,k,1,1),cull_mode=2) for k in range(x,x+width)]
        for cap in (2,7,16):
            actual=render.opaque_window(s,c,(64,x,1,width),cull_mode=2,tile_pixels=cap)
            for key in ('colour','mask'):
                expected=torch.cat([p[key] for p in canonical],2)
                assert torch.equal(actual[key].view(torch.uint8),expected.view(torch.uint8)),(key,x,width,cap)
