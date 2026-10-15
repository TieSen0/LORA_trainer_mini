"""Shared, non-training runtime probe; never installs packages or downloads models."""


def training_probe(config):
    probe = (
        "import sys, torch, accelerate, safetensors; "
        "print('Python:', sys.version.split()[0]); "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA build:', torch.version.cuda); "
        "print('Accelerate:', accelerate.__version__); "
        "assert torch.cuda.is_available(), 'A CUDA-capable PyTorch environment and GPU are required'; "
        "print('GPU:', torch.cuda.get_device_name(0)); "
    )
    if config.mixed_precision == "bf16":
        probe += "assert torch.cuda.is_bf16_supported(), 'Selected GPU does not support bf16'; "
    if config.attention_backend == "xformers":
        probe += "import xformers.ops; print('xFormers import: OK'); "
    if config.optimizer == "AdamW8bit":
        probe += "import bitsandbytes; print('bitsandbytes import: OK'); "
    return probe
