%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes triton
!pip install datasets sentencepiece protobuf

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  SEM GPU! Mude o Runtime para T4 GPU!")
