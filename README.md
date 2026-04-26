# ComfyUI Flash Attention V100

## Overview
A custom node for ComfyUI that patches the native attention mechanism to utilize Flash Attention optimized for NVIDIA Volta (V100) and Turing architecture GPUs. This extension enables accelerated inference for diffusion, video generation, and text-to-speech models on GPU architectures that lack native Flash Attention 2 support. It provides automatic tensor layout detection, numerical stability safeguards, and seamless fallback to PyTorch's native attention when necessary.

## Features
- Optimized attention execution for GPUs with compute capability less than 8.0
- Automatic detection and conversion of 3D and 4D tensor layouts from ComfyUI models
- Forced FP16 conversion to leverage Volta tensor cores
- Output sanitization to prevent NaN and Inf propagation in audio/video branches
- Graceful fallback to standard optimized attention on kernel failure
- Model type detection and configuration (checkpoint, diffusion, clip, ltxv, flux, qwen)
- Three dedicated ComfyUI nodes for control, monitoring, and configuration
- Full API compatibility with Dao-AILab flash-attention and transformers/Qwen3-TTS

## System Requirements
- NVIDIA GPU with compute capability less than 8.0 (Tesla V100, T4, RTX 20-series, Quadro RTX)
- CUDA 11.8 or newer
- Python 3.10 or newer
- ComfyUI (latest version recommended)
- [`flash_attn_v100` library](https://github.com/ai-bond/flash-attention-v100) installed in the environment

## Installation
1. Navigate to the ComfyUI `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   ```
2. Clone the repository:
   ```
   git clone git@github.com:NetVoobrazhenia/ComfyUI_Flash-Attention_v100.git
   ```
3. Install the required Volta-compatible Flash Attention library:
    
   Refer to the official documentation at https://github.com/ai-bond/flash-attention-v100 for build instructions.
4. Restart ComfyUI.

## Usage
After installation, three nodes will appear in the `attention/flash_v100` category:

### Flash Attn V100 Controller
Connect this node between your model loader and sampler/generator nodes.
- `enable_v100_opt`: Toggle the patch on or off
- `model_type`: Specify the architecture or leave as auto for automatic detection
- `model`: Connect the loaded model tensor
- `debug_mode`: Enable verbose console logging for troubleshooting

### Flash Attn V100 Status
A monitoring node that displays current GPU architecture, installation status of the underlying library, patch state, and detected model type. Connect to any text display node to view in the interface.

### Flash Attn V100 Config
Dynamically adjusts patch behavior without restarting ComfyUI.
- `force_fp16`: Converts all attention inputs to FP16 (required for Volta kernel execution)
- `sanitize_output`: Removes NaN and extreme values from attention outputs
- `sanitize_min` / `sanitize_max`: Bounds for output clamping
- `debug_mode`: Enables detailed logging

## Configuration
The patcher exposes a `PatchConfig` class that can be modified programmatically or via the Config node. Default values are optimized for stability on Volta hardware.

Environment variables:
- `FLASHATTN_V100_AUTO_PATCH=1`: Enables automatic patching on ComfyUI startup
- `COMFYUI_LOG_LEVEL=DEBUG`: Increases logging verbosity

## Limitations
- Input and output tensors must have head dimensions divisible by 8 due to Volta WMMA instruction requirements
- Dropout probability must be set to 0.0. Non-zero dropout is not supported by the Volta kernel
- ALiBi positional biases and softcap logit scaling are not implemented
- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) are not supported. Query and Key tensors must have identical head counts
- Performance gains are achieved through FP16 tensor core utilization. Models requiring BF16 or FP32 attention will incur conversion overhead
- The patch operates as a Python-level fallback when native variable-length CUDA kernels are unavailable. Processing is performed sequentially per sequence in batched scenarios

## Troubleshooting
- `RuntimeError: q must be fp16`: Ensure `force_fp16` is enabled. The Volta kernel strictly requires half-precision inputs
- `ValueError: shape is invalid for input of size`: Usually indicates a tensor layout mismatch or non-contiguous memory. Enable debug mode to inspect tensor shapes before the kernel call
- `No module named 'flash_attn.bert_padding'`: Verify that `flash_attn_v100` is installed correctly. The patch includes a built-in fallback implementation if the module is missing
- Audio/Video generation produces corrupted output: Enable `sanitize_output` in the Config node. This prevents numerical instability in long-sequence softmax operations
- Performance degradation: Disable the patch for models that natively support SM_80+ architectures. The patch is designed exclusively for compute capability less than 8.0

## License
BSD-3-Clause
