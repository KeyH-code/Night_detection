from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_style_transfers_constant_low_frequency_brightness_only_inside_source_content():
    from night_style import source_style

    v=torch.full((2,3,16,16),.45)
    v[0,:,4:12]=.75
    v[1,:,4:12]=.25
    out=source_style(v,slice(4,12),.05,.8)
    torch.testing.assert_close(out[0,:,4:12],torch.full((3,8,16),.35))
    assert torch.equal(out[1],v[1])
    assert torch.equal(out[0,:,:4],v[0,:,:4])
    assert torch.equal(out[0,:,12:],v[0,:,12:])
    assert (v[0,:,4:12]==.75).all()


def test_style_preserves_untransferred_high_frequencies_without_clipping():
    from night_style import source_style

    v=torch.ones(2,3,16,16)*.5
    high = .08*torch.cos(2*torch.pi*torch.arange(16)*4/16)
    v[0]+=high
    v[1]-=.1
    out=source_style(v,slice(0,16),.07,.8)
    torch.testing.assert_close(out[0]-out[0].mean(),v[0]-v[0].mean(),atol=2e-7,rtol=1e-5)
