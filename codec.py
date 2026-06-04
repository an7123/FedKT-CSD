"""DC-AE encoder/decoder wrapper.

The Deep Compression AutoEncoder (DC-AE) is pretrained on ImageNet and
provides a 32x spatial compression: 64x64 RGB -> (32, 2, 2) = 128-dim latent.
Both encoder and decoder are frozen throughout -- no privacy cost.

DC-AE lives in mit-han-lab/efficientvit, which is not on PyPI. setup.sh
clones it next to this file (or override with EFFICIENTVIT_DIR=...).
Note: DC-AE uses Triton kernels and must run on cuda:0.
"""

import importlib.machinery
import os
import sys
import types
import torch
import torch.nn as nn

# Mock optional ONNX export deps (not needed for inference). The upstream
# efficientvit package imports onnx eagerly through apps.utils.export, but
# we only need the DC-AE model definition.
for _name in ("onnx", "onnxsim"):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.simplify = lambda x, **kw: (x, True)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, None)
        sys.modules[_name] = _m


def _locate_efficientvit() -> str:
    """Find the efficientvit checkout. Search order:
       1. $EFFICIENTVIT_DIR
       2. <this file>/efficientvit  (vendored alongside the minimal repo)
       3. /tmp/efficientvit          (default setup.sh target)
    """
    candidates = [
        os.environ.get("EFFICIENTVIT_DIR"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientvit"),
        "/tmp/efficientvit",
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "efficientvit")):
            return c
    raise RuntimeError(
        "efficientvit checkout not found. Run ./setup.sh or set "
        "EFFICIENTVIT_DIR to a clone of github.com/mit-han-lab/efficientvit."
    )


sys.path.insert(0, _locate_efficientvit())
from efficientvit.ae_model_zoo import DCAE_HF


class Encoder(nn.Module):
    def __init__(self, dc_ae):
        super().__init__()
        self.dc_ae = dc_ae

    def forward(self, x):
        return self.dc_ae.encode(x * 2 - 1)          # [0,1] -> [-1,1]


class Decoder(nn.Module):
    def __init__(self, dc_ae):
        super().__init__()
        self.dc_ae = dc_ae

    def forward(self, z):
        return (self.dc_ae.decode(z) + 1) / 2         # [-1,1] -> [0,1]


def load_dcae(model_name="dc-ae-f32c32-in-1.0", device="cuda:0"):
    """Load pretrained DC-AE.  Returns (encoder, decoder), both frozen."""
    dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/{model_name}")
    dc_ae = dc_ae.to(device).eval()
    enc = Encoder(dc_ae).eval()
    dec = Decoder(dc_ae).eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    return enc, dec
