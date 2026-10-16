"""Separate the retained soft fitting model from opaque image acceptance."""
import importlib
import torch
from test_deferred import fixture


def test_legacy_soft_fitting_model_exposes_documented_interior_seam():
    model,camera,*_=fixture()
    d=importlib.import_module('gffx.torch._deferred')
    hard=d.render(model,camera,outputs=('mask',))['mask']
    soft=d.render(model,camera,soft=.2,outputs=('mask',))['mask']
    assert torch.equal(hard,torch.ones_like(hard))
    assert bool((soft<.75).any()),'diagnostic fixture no longer reproduces the historical seam'


def test_opaque_coverage_mode_has_no_shared_diagonal_seam():
    # New model acceptance, expected red until its implementation is qualified.
    model,camera,*_=fixture()
    d=importlib.import_module('gffx.torch._deferred')
    opaque=d.render(model,camera,coverage='opaque',outputs=('mask',))['mask']
    assert torch.equal(opaque,torch.ones_like(opaque))
