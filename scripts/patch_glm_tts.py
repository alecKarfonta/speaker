#!/usr/bin/env python3
"""
Patch GLM-TTS source code for:
1. Dtype consistency (fixes FP32/FP16 matmul errors)
2. Performance optimizations

Applied during Docker build after cloning GLM-TTS.
"""

import os
import sys
import re

def patch_file(filepath: str, patches: list) -> bool:
    """Apply patches to a file. Returns True if any patches applied."""
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for name, old, new in patches:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ {name}")
        else:
            print(f"  ⚠ {name} (already patched or different)")
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    glm_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/GLM-TTS"
    
    print("=" * 50)
    print("Patching GLM-TTS for dtype & performance")
    print("=" * 50)
    
    # =========================================================================
    # PATCH 1: flow/modules.py - Fix dtype in embeddings
    # =========================================================================
    print("\n[PATCH 1] flow/modules.py - Fixing dtype consistency...")
    
    modules_patches = [
        # Patch 1a: SinusPositionEmbedding - use input tensor dtype
        (
            "SinusPositionEmbedding.forward() dtype fix",
            '''    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb''',
            '''    def forward(self, x, scale=1000):
        device = x.device
        # Use input dtype for consistency (fixes FP32/FP16 mismatch)
        dtype = x.dtype if x.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = scale * x.unsqueeze(1).to(dtype) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb'''
        ),
        # Patch 1b: TimestepEmbedding - cast to layer dtype before MLP
        (
            "TimestepEmbedding.forward() dtype fix",
            '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden) 
        return time''',
            '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        # Ensure timestep is float for sin/cos embedding computation
        if not timestep.is_floating_point():
            timestep = timestep.float()
        time_hidden = self.time_embed(timestep)
        # Cast to layer dtype to prevent FP32/FP16 mismatch in time_mlp
        time_hidden = time_hidden.to(self.time_mlp[0].weight.dtype)
        time = self.time_mlp(time_hidden)
        return time'''
        ),
    ]
    
    patch_file(os.path.join(glm_dir, "flow/modules.py"), modules_patches)
    
    # =========================================================================
    # PATCH 2: flow/flow.py - Fix all FP32 tensor creations
    # =========================================================================
    print("\n[PATCH 2] flow/flow.py - Fixing tensor dtype consistency...")
    
    flow_patches = [
        # Patch 2a: mel_cond_btd should use prompt_feat.dtype (ROOT CAUSE)
        (
            "inference_with_cache() mel_cond_btd dtype fix",
            "mel_cond_btd = torch.zeros([1, feat_len, self.mel_dim]).to(device)",
            "mel_cond_btd = torch.zeros([1, feat_len, self.mel_dim], device=device, dtype=prompt_feat.dtype)"
        ),
        # Patch 2b: linspace should use model dtype  
        (
            "do_sample() timestep dtype fix",
            "t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)",
            "t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mel_cond_btd.dtype)"
        ),
    ]
    
    patch_file(os.path.join(glm_dir, "flow/flow.py"), flow_patches)
    
    # =========================================================================
    # VERIFY: Check that SDPA is being used
    # =========================================================================
    print("\n[VERIFY] Checking for SDPA (scaled_dot_product_attention)...")
    modules_path = os.path.join(glm_dir, "flow/modules.py")
    if os.path.exists(modules_path):
        with open(modules_path) as f:
            if "scaled_dot_product_attention" in f.read():
                print("  ✓ SDPA already in use (good for performance)")
            else:
                print("  ⚠ SDPA not found - using manual attention")
    
    print("\n" + "=" * 50)
    print("GLM-TTS patching complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
