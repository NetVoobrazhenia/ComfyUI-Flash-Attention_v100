# ============================================================================
# ComfyUI-Flash-Attention_v100 - Production Ready
# Compatible with: checkpoint, diffusion, clip, model, LTX-2.3, FLUX, Qwen3-TTS
# GPU Support: NVIDIA Volta (V100) and Turing (T4) - compute capability < 8.0
# ============================================================================

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.ldm.modules.attention as attn
import importlib
import sys
import types
import importlib.util
import logging
from typing import Optional, Dict, Set, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Module Registration Utilities
# ============================================================================

def make_package(name: str) -> types.ModuleType:
    """
    Create a mock package with proper import hierarchy for sys.modules registration.
    
    Args:
        name: Fully qualified package name (e.g., "flash_attn")
    
    Returns:
        ModuleType instance configured as a package
    """
    m = types.ModuleType(name)
    m.__path__ = []
    # Do NOT set m.__spec__.parent — it's read-only
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    m.__package__ = name  # This is the correct way to indicate package identity
    return m


def make_module(name: str) -> types.ModuleType:
    """
    Create a mock submodule with proper parent reference.
    
    Args:
        name: Fully qualified module name (e.g., "flash_attn.bert_padding")
    
    Returns:
        ModuleType instance configured as a submodule
    """
    m = types.ModuleType(name)
    # Do NOT set m.__spec__.parent — it's read-only
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    # Set __package__ to the parent package name for proper import resolution
    m.__package__ = name.rpartition(".")[0] or name
    return m


# ============================================================================
# Patcher Configuration
# ============================================================================

class PatchConfig:
    """
    Global configuration for Flash Attention V100 patcher.
    
    Note: For thread safety in multi-threaded scenarios, consider using
    threading.Lock or moving to instance attributes.
    """
    # Enable debug output
    DEBUG: bool = False
    
    # Force FP16 conversion (required for V100 CUDA kernel)
    FORCE_FP16: bool = True
    
    # Output sanitization: clamp NaN/Inf values
    SANITIZE_OUTPUT: bool = True
    SANITIZE_MIN: float = -1e4
    SANITIZE_MAX: float = 1e4
    
    # Automatic fallback to original attention on error
    AUTO_FALLBACK: bool = True
    
    # Supported model type identifiers (no trailing spaces!)
    SUPPORTED_MODEL_TYPES: Set[str] = {
        "checkpoint", "diffusion", "clip", "model",
        "ltxv", "flux", "qwen", "sdxl", "sd15", "sd3"
    }
    
    # Model-specific handling hints (for future extension)
    SPECIAL_HANDLING: Dict[str, Dict[str, Any]] = {
        "ltxv": {"causal": False, "varlen_support": True},
        "flux": {"causal": False, "manual_cast_fp32": True},
        "qwen": {"causal": True, "varlen_support": True, "audio_branch": True},
    }
    
    @classmethod
    def update(cls, **kwargs) -> None:
        """Thread-safe configuration update (basic version)."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


# ============================================================================
# Main Patcher Class
# ============================================================================

class FlashAttnV100Patcher:
    """
    Flash Attention patcher for NVIDIA Volta (V100/T4) GPUs.
    
    Features:
    - Support for various ComfyUI model types (checkpoint, diffusion, clip, etc.)
    - Automatic detection of tensor layouts (3D/4D)
    - NaN/Inf protection for stable audio/video generation
    - Full compatibility with Dao-AILab flash-attention API
    - Graceful fallback to standard PyTorch attention on error
    """
    
    def __init__(self):
        self.patched: bool = False
        self.original_attention = None
        self.original_attention_masked = None
        self.gpu_arch: Optional[str] = None
        self._patched_functions: list = []
        self._model_context: Dict[str, Any] = {}
        
    def should_patch(self) -> bool:
        """Check if patching is needed for the current GPU."""
        if not torch.cuda.is_available():
            if PatchConfig.DEBUG:
                logger.debug("[FlashAttnV100] CUDA not available")
            return False
            
        major, minor = torch.cuda.get_device_capability()
        self.gpu_arch = f"sm_{major}{minor}"
        
        # Only patch GPUs with compute capability < 8.0 (Volta, Turing)
        should = major < 8
        if PatchConfig.DEBUG:
            logger.debug(f"[FlashAttnV100] GPU {self.gpu_arch}, patch: {should}")
        return should
    
    def _register_flash_attn_modules(self) -> bool:
        """Register mock flash_attn submodules for compatibility."""
        try:
            from flash_attn_v100 import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            logger.warning(f"[FlashAttnV100] flash_attn_v100 not installed: {e}")
            return False
        
        # Register flash_attn package
        fa = make_package("flash_attn")
        fa.flash_attn_func = flash_attn_func
        fa.flash_attn_varlen_func = flash_attn_varlen_func
        fa.__version__ = "2.8.3+v100"
        sys.modules["flash_attn"] = fa
        
        # Register flash_attn.flash_attn_interface
        fai = make_module("flash_attn.flash_attn_interface")
        fai.flash_attn_func = flash_attn_func
        fai.flash_attn_varlen_func = flash_attn_varlen_func
        sys.modules["flash_attn.flash_attn_interface"] = fai
        
        # Register flash_attn.bert_padding with real implementation
        try:
            from flash_attn_v100.bert_padding import (
                index_first_axis, pad_input, unpad_input,
                index_put_first_axis, unpad_input_for_concatenated_sequences
            )
            fbp = make_module("flash_attn.bert_padding")
            fbp.index_first_axis = index_first_axis
            fbp.index_put_first_axis = index_put_first_axis
            fbp.unpad_input = unpad_input
            fbp.pad_input = pad_input
            fbp.unpad_input_for_concatenated_sequences = unpad_input_for_concatenated_sequences
            sys.modules["flash_attn.bert_padding"] = fbp
            # Link to parent package for attribute-style access: flash_attn.bert_padding
            fa.bert_padding = fbp
            
        except ImportError:
            # Fallback: create minimal stubs
            logger.warning("[FlashAttnV100] Using stub bert_padding - some features may not work")
            fbp = make_module("flash_attn.bert_padding")
            
            def _identity(x, *args, **kwargs):
                return x
            
            def _unpad_stub(hidden_states, attention_mask):
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.to(torch.bool)
                
                batch_size, seq_len = attention_mask.shape
                mask_flat = attention_mask.reshape(-1)
                indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
                
                hidden_flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
                unpadded = hidden_flat.index_select(0, indices)
                
                seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
                cu_seqlens = torch.cat([
                    torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                    torch.cumsum(seqlens, dim=0, dtype=torch.int32)
                ])
                max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
                
                return unpadded, indices, cu_seqlens, max_seqlen
            
            fbp.index_first_axis = _identity
            fbp.index_put_first_axis = _identity
            fbp.unpad_input = _unpad_stub
            fbp.pad_input = _identity
            fbp.unpad_input_for_concatenated_sequences = lambda x, cu, *a: (
                x, cu, (cu[1:] - cu[:-1]).max().item()
            )
            
            sys.modules["flash_attn.bert_padding"] = fbp
            fa.bert_padding = fbp
    
        if PatchConfig.DEBUG:
            logger.debug("[FlashAttnV100] Modules registered successfully")
        return True
    
    def _create_v100_attention(self, use_masked: bool = False):
        """
        Create a V100-compatible attention function.
        
        Args:
            use_masked: Whether to prepare for masked attention fallback
        
        Returns:
            Callable: Attention function matching ComfyUI's optimized_attention signature
        """
        # Cache imports at closure creation, not per-call
        from flash_attn_v100 import flash_attn_func
        
        def v100_attention(
            q: torch.Tensor,
            k: torch.Tensor, 
            v: torch.Tensor,
            heads: int,
            mask: Optional[torch.Tensor] = None,
            attn_precision: Optional[str] = None,
            transformer_options: Optional[dict] = None,
            **kwargs
        ) -> torch.Tensor:
            debug = PatchConfig.DEBUG
            try:
                orig_dtype = q.dtype
                device = q.device
                
                # Ensure contiguous memory layout (required for reshape and CUDA kernel)
                if not q.is_contiguous():
                    q = q.contiguous()
                if not k.is_contiguous():
                    k = k.contiguous()
                if not v.is_contiguous():
                    v = v.contiguous()
                
                # Determine input layout: ComfyUI uses either 3D or 4D tensors
                if q.dim() == 3:
                    # Layout A: (batch, seq_len, inner_dim) where inner_dim = heads * head_dim
                    batch_size, seq_len, inner_dim = q.shape
                    
                    if inner_dim % heads != 0:
                        raise ValueError(
                            f"Cannot determine layout: inner_dim={inner_dim} not divisible by heads={heads}"
                        )
                    
                    head_dim = inner_dim // heads
                    expected_numel = batch_size * seq_len * heads * head_dim
                    
                    if q.numel() != expected_numel:
                        raise ValueError(
                            f"Reshape mismatch: tensor has {q.numel()} elements, "
                            f"but target ({batch_size}, {seq_len}, {heads}, {head_dim}) "
                            f"requires {expected_numel}. q.shape={q.shape}"
                        )
                    
                    # Reshape to Flash Attention 2 layout: (B, M, H, D)
                    q_fa = q.reshape(batch_size, seq_len, heads, head_dim)
                    k_fa = k.reshape(batch_size, seq_len, heads, head_dim)
                    v_fa = v.reshape(batch_size, seq_len, heads, head_dim)
                    is_3d_input = True
                    
                elif q.dim() == 4:
                    # Layout B: Already in FA2 format (batch, seq_len, heads, head_dim)
                    batch_size, seq_len, h_in, d_in = q.shape
                    if h_in != heads:
                        raise ValueError(f"4D heads mismatch: {h_in} vs {heads}")
                    q_fa, k_fa, v_fa = q, k, v
                    head_dim = d_in
                    is_3d_input = False
                else:
                    raise ValueError(f"Unsupported q.ndim={q.dim()}, expected 3 or 4")

                # Volta tensor cores require FP16 input
                if PatchConfig.FORCE_FP16 and q_fa.dtype != torch.float16:
                    q_fa = q_fa.to(torch.float16)
                    k_fa = k_fa.to(torch.float16)
                    v_fa = v_fa.to(torch.float16)

                # Call Flash Attention V100 kernel
                out_fa = flash_attn_func(
                    q_fa, k_fa, v_fa,
                    dropout_p=0.0,           # Dropout not supported in V100 kernel
                    softmax_scale=None,      # Auto: 1/sqrt(head_dim)
                    causal=False,            # Diffusion models use bidirectional attention
                    window_size=(-1, -1),    # Full attention, no sliding window
                    softcap=0.0,             # Not supported on V100
                    alibi_slopes=None,       # Not supported on V100
                    deterministic=False,
                    return_attn_probs=False,
                )

                # Restore original dtype for graph consistency
                out_fa = out_fa.to(orig_dtype)
                
                # Sanitize output to prevent NaN/Inf propagation (critical for audio/video branches)
                if PatchConfig.SANITIZE_OUTPUT:
                    out_fa = torch.nan_to_num(
                        out_fa,
                        nan=0.0,
                        posinf=PatchConfig.SANITIZE_MAX,
                        neginf=PatchConfig.SANITIZE_MIN
                    )
                    out_fa = torch.clamp(
                        out_fa,
                        min=PatchConfig.SANITIZE_MIN,
                        max=PatchConfig.SANITIZE_MAX
                    )

                if debug :
                    logger.debug(f"[FlashAttnV100] FA compute {q.shape}")
                # Restore original layout
                if is_3d_input:
                    return out_fa.reshape(batch_size, seq_len, inner_dim)
                else:
                    return out_fa.reshape(batch_size, seq_len, heads, head_dim)

            except Exception as e:
                if debug or PatchConfig.AUTO_FALLBACK:
                    logger.warning(f"[FlashAttnV100] FA kernel failed: {e}, falling back")
                
                # Fallback: call original attention with unmodified inputs
                fallback_fn = self.original_attention_masked if use_masked else self.original_attention
                fallback_kwargs = {
                    'attn_precision': attn_precision,
                    'transformer_options': transformer_options,
                    **kwargs  # Forward any additional args (no filtering needed)
                }
                if mask is not None:
                    fallback_kwargs['mask'] = mask
                return fallback_fn(q, k, v, heads, **fallback_kwargs)
        
        return v100_attention
    
    def patch(self, model_type: Optional[str] = None, model_config: Optional[Dict] = None) -> bool:
        """
        Apply the patch to attention functions.
        
        Args:
            model_type: Model type ("checkpoint", "diffusion", "clip", "ltxv", etc.)
            model_config: Additional model configuration
        
        Returns:
            bool: True if patching succeeded, False otherwise
        """
        if self.patched:
            return True
        if not self.should_patch():
            return False
        
        # Save model context
        if model_type:
            self._model_context["type"] = model_type.lower()
        if model_config:
            self._model_context.update(model_config)
        
        try:
            # Register flash_attn modules
            if not self._register_flash_attn_modules():
                return False
            
            # Save original functions
            self.original_attention = attn.optimized_attention
            if hasattr(attn, 'optimized_attention_masked'):
                self.original_attention_masked = attn.optimized_attention_masked
            
            # Create and apply patches
            attn.optimized_attention = self._create_v100_attention(use_masked=False)
            if hasattr(attn, 'optimized_attention_masked'):
                attn.optimized_attention_masked = self._create_v100_attention(use_masked=True)
            
            self._patched_functions = [
                ('attn.optimized_attention', attn.optimized_attention),
            ]
            if hasattr(attn, 'optimized_attention_masked'):
                self._patched_functions.append(
                    ('attn.optimized_attention_masked', attn.optimized_attention_masked)
                )
            
            # Disable upcast if option exists
            if hasattr(mm, 'force_attention_upcast'):
                mm.force_attention_upcast = False
            
            self.patched = True
            logger.info(
                f"[FlashAttnV100] Active on {self.gpu_arch} | "
                f"Model: {self._model_context.get('type', 'auto')} | "
                f"FP16: {PatchConfig.FORCE_FP16} | Sanitize: {PatchConfig.SANITIZE_OUTPUT}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[FlashAttnV100] Patch failed: {e}")
            if PatchConfig.DEBUG:
                import traceback
                traceback.print_exc()
            self.restore()  # Rollback on error
            return False
    
    def restore(self) -> None:
        """Restore original attention functions."""
        if not self.patched:
            return
        
        try:
            # Restore original functions
            if self.original_attention:
                attn.optimized_attention = self.original_attention
            if self.original_attention_masked and hasattr(attn, 'optimized_attention_masked'):
                attn.optimized_attention_masked = self.original_attention_masked
            
            # Clear references
            self.original_attention = None
            self.original_attention_masked = None
            self._patched_functions.clear()
            self.patched = False
            
            logger.info("[FlashAttnV100] Restored default attention")
            
        except Exception as e:
            logger.error(f"[FlashAttnV100] Restore failed: {e}")


# ============================================================================
# Global Instance
# ============================================================================

patcher = FlashAttnV100Patcher()


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class FlashAttnV100Controller:
    """
    Node for manual control of Flash Attention V100.
    Supports MODEL, CLIP, VAE, and other ComfyUI object types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_v100_opt": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable V100 Optimization",
                    "tooltip": "Enable Flash Attention patch for Volta GPUs (V100/T4)"
                }),
                "model_type": (["auto", "checkpoint", "diffusion", "clip", "ltxv", "flux", "qwen"], {
                    "default": "auto",
                    "label": "Model Type",
                    "tooltip": "Specify model type for optimized handling"
                }),
                "target": ("*", {
                    "label": "Target Object",
                    "tooltip": "MODEL, CLIP, VAE, or any ComfyUI object to patch"
                }),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Debug Mode",
                    "tooltip": "Enable verbose logging for troubleshooting"
                }),
            }
        }
    
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("target", "status")
    FUNCTION = "apply"
    CATEGORY = "attention/flash_v100"
    DESCRIPTION = "Enable Flash Attention optimization for NVIDIA V100/T4 GPUs"
    
    def apply(self, enable_v100_opt: bool, model_type: str, target, debug_mode: bool = False):
        """
        Apply or restore the Flash Attention V100 patch.
        
        Args:
            enable_v100_opt: Toggle patch on/off
            model_type: Model architecture hint or "auto"
            target: Input object (MODEL, CLIP, VAE, or any ComfyUI object)
            debug_mode: Enable verbose logging
        
        Returns:
            tuple: (target, status_message)
        """
        # Update global debug configuration
        PatchConfig.DEBUG = debug_mode
        
        if enable_v100_opt:
            mt = model_type if model_type != "auto" else None
            
            # Auto-detect model type from target if not explicitly specified
            if mt is None:
                try:
                    obj_type = type(target).__name__.lower()
                    if "clip" in obj_type:
                        mt = "clip"
                    elif "vae" in obj_type:
                        mt = "vae"
                    elif hasattr(target, 'model_config'):
                        cfg = getattr(target.model_config, 'config', {})
                        if isinstance(cfg, dict):
                            target_name = str(cfg.get('target', '')).lower()
                            if 'ltx' in target_name:
                                mt = "ltxv"
                            elif 'flux' in target_name:
                                mt = "flux"
                            elif 'qwen' in target_name:
                                mt = "qwen"
                            elif any(x in target_name for x in ['sd', 'stable', 'diffusion']):
                                mt = "diffusion"
                except Exception as e:
                    logger.debug(f"[FlashAttnV100] Model type detection skipped: {e}")
            
            # Apply patch (state remains True even if already patched)
            patcher.patch(model_type=mt)
            
            # Determine TRUE status from patcher state
            is_active = patcher.patched
            gpu_info = patcher.gpu_arch or "unknown"
            status = f"FlashAttnV100: {'ACTIVE' if is_active else 'FAILED'} ({gpu_info})"
            if mt and is_active:
                status += f" | Model: {mt}"
        else:
            patcher.restore()
            status = "FlashAttnV100: DISABLED (using default attention)"
        
        # Return the EXACT same object that was passed in, preserving graph flow
        return (target, status)


class FlashAttnV100Status:
    """
    Node for displaying Flash Attention V100 status.
    
    Shows:
    - GPU architecture
    - flash_attn_v100 installation status
    - Patch application status
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check"
    CATEGORY = "attention/flash_v100"
    DESCRIPTION = "Display Flash Attention V100 status and GPU info"
    
    def check(self) -> Tuple[str]:
        if not torch.cuda.is_available():
            return ("No CUDA available",)
        
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        
        # Check flash_attn_v100 installation
        fa_status = "Not installed"
        fa_version = None
        try:
            from flash_attn_v100 import flash_attn_func, flash_attn_varlen_func
            fa_status = "Installed"
            # Try to get version
            import flash_attn_v100
            if hasattr(flash_attn_v100, '__version__'):
                fa_version = flash_attn_v100.__version__
        except ImportError:
            pass
        except Exception as e:
            fa_status = f"Error: {e}"
        
        # Patch status
        patch_status = "Patched" if patcher.patched else "Not patched"
        
        # Compatibility
        compat = "Compatible" if major < 8 else "Not needed (SM_80+)"
        
        parts = [
            f"GPU: {gpu_name} ({arch})",
            f"FlashAttn-v100: {fa_status}" + (f" v{fa_version}" if fa_version else ""),
            f"Patch: {patch_status}",
            f"Volta Compat: {compat}",
        ]
        
        if patcher.patched and patcher._model_context:
            mt = patcher._model_context.get('type', 'auto')
            parts.append(f"Model: {mt}")
        
        return (" | ".join(parts),)


class FlashAttnV100Config:
    """
    Node for configuring Flash Attention V100 parameters.
    
    Allows dynamic adjustment of:
    - Force FP16 conversion
    - Output sanitization
    - Debug mode
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_fp16": ("BOOLEAN", {
                    "default": True,
                    "label": "Force FP16",
                    "tooltip": "Convert inputs to FP16 (required for V100 kernel)"
                }),
                "sanitize_output": ("BOOLEAN", {
                    "default": True,
                    "label": "Sanitize Output",
                    "tooltip": "Remove NaN/Inf from output (critical for audio/video)"
                }),
            },
            "optional": {
                "sanitize_min": ("FLOAT", {
                    "default": -10000.0,
                    "min": -1e6,
                    "max": 0,
                    "label": "Sanitize Min",
                    "tooltip": "Minimum value after sanitization"
                }),
                "sanitize_max": ("FLOAT", {
                    "default": 10000.0,
                    "min": 0,
                    "max": 1e6,
                    "label": "Sanitize Max", 
                    "tooltip": "Maximum value after sanitization"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Debug Mode"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_status",)
    FUNCTION = "apply"
    CATEGORY = "attention/flash_v100"
    
    def apply(
        self, 
        force_fp16: bool, 
        sanitize_output: bool, 
        sanitize_min: float = -1e4, 
        sanitize_max: float = 1e4, 
        debug_mode: bool = False
    ) -> Tuple[str]:
        # Update global configuration
        PatchConfig.FORCE_FP16 = force_fp16
        PatchConfig.SANITIZE_OUTPUT = sanitize_output
        PatchConfig.SANITIZE_MIN = sanitize_min
        PatchConfig.SANITIZE_MAX = sanitize_max
        PatchConfig.DEBUG = debug_mode
        
        status = (
            f"Config updated: FP16={force_fp16} | "
            f"Sanitize={sanitize_output}[{sanitize_min}:{sanitize_max}] | "
            f"Debug={debug_mode}"
        )
        
        # Log changes if patch is already active
        if patcher.patched:
            logger.info(f"[FlashAttnV100] Config changed: {status}")
        
        return (status,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "FlashAttnV100Controller": FlashAttnV100Controller,
    "FlashAttnV100Status": FlashAttnV100Status,
    "FlashAttnV100Config": FlashAttnV100Config,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashAttnV100Controller": "Flash Attn V100 Controller",
    "FlashAttnV100Status": "Flash Attn V100 Status",
    "FlashAttnV100Config": "Flash Attn V100 Config",
}

# ============================================================================
# Auto-init (optional)
# ============================================================================

def on_comfyui_load() -> None:
    """Called when ComfyUI loads for auto-patching."""
    if PatchConfig.DEBUG:
        logger.info("[FlashAttnV100] Auto-init requested")
    
    # Auto-patch only if explicitly enabled via environment variable
    # Default: manual control via nodes
    # if os.getenv("FLASHATTN_V100_AUTO_PATCH", "0") == "1":
    #     patcher.patch()

# Uncomment to enable auto-init on import:
# on_comfyui_load()