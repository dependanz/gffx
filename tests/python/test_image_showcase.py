"""Finite image demonstration structure; visual inspection remains an independent requirement."""
import torch


def test_image_hero_uses_public_layer_operations(monkeypatch):
    """The poster must actually exercise gffx compositing/warping/effects rather than embed an external render."""
    import gffx.image as image
    from tools.testing_dashboard.image_showcase import hero
    calls={name:0 for name in ('composite','warp_affine','gaussian_blur','adjust','apply_mask')}
    for name in calls:
        original=getattr(image,name)
        def tracked(*args,_name=name,_original=original,**kwargs):
            calls[_name]+=1
            return _original(*args,**kwargs)
        monkeypatch.setattr(image,name,tracked)
    result=hero((180,320))
    assert result.shape==(180,320,3) and result.dtype==torch.uint8
    assert all(count>0 for count in calls.values())
    assert int(result.max())-int(result.min())>150


def test_image_publication_is_finite_and_honestly_labelled():
    """A static presentation uses bounded images/text, not a running producer or numerical acceptance claim."""
    from tools.testing_dashboard.image_showcase import publish_data
    calls=[]
    class Sink:
        def image(self,*args,**kwargs): calls.append(('image',args,kwargs))
        def text(self,*args,**kwargs): calls.append(('text',args,kwargs))
    images=[torch.zeros(4,8,3,dtype=torch.uint8)]*3
    artifacts=publish_data(Sink(),images)
    assert len(artifacts)==4 and len(calls)==4
    assert [row[0] for row in calls].count('image')==3
    guide=' '.join(str(row[1]) for row in calls if row[0]=='text')
    assert 'not test acceptance' in guide and 'authored' in guide


def test_showcase_labels_have_all_required_glyphs():
    """The inspected first preview lost its factor-four label because the fixture font lacked digit4."""
    from tools.testing_dashboard.image_showcase import _FONT
    labels='GFFX / IMAGE LAB CHROMA LINEAR LIGHT LAYERS MASKS WARP 01 PUBLIC IMAGE OPERATIONS 4X BOX RESOLVE'
    assert set(labels)-{' '} <= set(_FONT)
