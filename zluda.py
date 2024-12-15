"""
ZLUDA emulation support for AMD. This script hijacks pytorch features not fully supported by Zluda and runs them on CPU instead.
torch.stft

  torch.stft - input and window have to be moved to CPU to prevent an exception
  torch.jit.script - wrapper needs to be unwrapped to prevent an exception

NYI
  torch.istft
  abs(complex64)
  pow(complex64)
"""

import torch

if torch.cuda.is_available() and torch.cuda.get_device_name().endswith("[ZLUDA]"):
    _torch_stft = torch.stft

    def z_stft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        return _torch_stft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(
            input.device
        )

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    # hijacks
    torch.stft = z_stft
    torch.jit.script = z_jit
    # disabling unsupported cudnn
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
