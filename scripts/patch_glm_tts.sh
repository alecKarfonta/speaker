#!/bin/bash
# Comprehensive GLM-TTS patches for:
# 1. Dtype consistency (fixes FP32/FP16 matmul errors)
# 2. Performance optimizations
#
# These patches are applied after cloning GLM-TTS in the Docker build.

set -e

GLM_TTS_DIR="${1:-/app/GLM-TTS}"

echo "========================================"
echo "Patching GLM-TTS for dtype & performance"
echo "========================================"

# =============================================================================
# PATCH 1: Fix dtype in SinusPositionEmbedding (flow/modules.py)
# Issue: Creates embeddings as .float() which causes FP32/FP16 mismatch
# Fix: Use input tensor's dtype instead of hardcoded float()
# =============================================================================
MODULES_FILE="$GLM_TTS_DIR/flow/modules.py"

if [ -f "$MODULES_FILE" ]; then
    echo "[PATCH 1] Fixing dtype in SinusPositionEmbedding..."
    
    # Create a Python script for more complex patching
    python3 << 'PATCH_SCRIPT'
import re
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/app/GLM-TTS/flow/modules.py"

with open(filepath, 'r') as f:
    content = f.read()

# Patch 1a: SinusPositionEmbedding.forward() - use input dtype
# Original: emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
# Fixed: Use x.dtype for the embedding creation
old_sinus = '''    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb'''

new_sinus = '''    def forward(self, x, scale=1000):
        device = x.device
        dtype = x.dtype if x.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = scale * x.unsqueeze(1).to(dtype) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb'''

if old_sinus in content:
    content = content.replace(old_sinus, new_sinus)
    print("  ✓ Patched SinusPositionEmbedding.forward()")
else:
    print("  ⚠ SinusPositionEmbedding already patched or different")

# Patch 1b: TimestepEmbedding.forward() - ensure dtype consistency
# The time_mlp output should match input dtype
old_timestep = '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden) 
        return time'''

new_timestep = '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        # Ensure timestep is float for embedding computation
        if not timestep.is_floating_point():
            timestep = timestep.float()
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)
        return time'''

if old_timestep in content:
    content = content.replace(old_timestep, new_timestep)
    print("  ✓ Patched TimestepEmbedding.forward()")
else:
    print("  ⚠ TimestepEmbedding already patched or different")

with open(filepath, 'w') as f:
    f.write(content)

print(f"  Saved: {filepath}")
PATCH_SCRIPT
    python3 -c "import sys; exec(open('/dev/stdin').read())" "$MODULES_FILE" << 'PATCH_SCRIPT'
import re
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/app/GLM-TTS/flow/modules.py"

with open(filepath, 'r') as f:
    content = f.read()

# Patch 1a: SinusPositionEmbedding.forward() - use input dtype
old_sinus = '''    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb'''

new_sinus = '''    def forward(self, x, scale=1000):
        device = x.device
        dtype = x.dtype if x.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = scale * x.unsqueeze(1).to(dtype) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb'''

if old_sinus in content:
    content = content.replace(old_sinus, new_sinus)
    print("  ✓ Patched SinusPositionEmbedding.forward()")
else:
    print("  ⚠ SinusPositionEmbedding already patched or different")

# Patch 1b: TimestepEmbedding.forward() - ensure dtype consistency  
old_timestep = '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden) 
        return time'''

new_timestep = '''    def forward(self, timestep: torch.Tensor):
        """
        Args:
            timestep: Tensor of shape (batch,)
        Returns:
            Time embedding of shape (batch, dim)
        """
        # Ensure timestep is float for embedding computation
        if not timestep.is_floating_point():
            timestep = timestep.float()
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)
        return time'''

if old_timestep in content:
    content = content.replace(old_timestep, new_timestep)
    print("  ✓ Patched TimestepEmbedding.forward()")
else:
    print("  ⚠ TimestepEmbedding already patched or different")

with open(filepath, 'w') as f:
    f.write(content)

print(f"  Saved: {filepath}")
PATCH_SCRIPT
    echo "  Done: $MODULES_FILE"
else
    echo "  Warning: $MODULES_FILE not found"
fi

# =============================================================================
# PATCH 2: flow/flow.py - Ensure timestep tensor uses correct dtype
# Issue: t_current is created as float32 via torch.linspace
# Fix: Cast to model dtype before passing to DiT
# =============================================================================
FLOW_FILE="$GLM_TTS_DIR/flow/flow.py"

if [ -f "$FLOW_FILE" ]; then
    echo "[PATCH 2] Fixing timestep dtype in flow.py..."
    
    # Replace linspace to use same dtype as mel_cond_btd
    sed -i 's/t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)/t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=mel_cond_btd.dtype)/g' "$FLOW_FILE"
    
    echo "  Done: $FLOW_FILE"
else
    echo "  Warning: $FLOW_FILE not found"
fi

# =============================================================================
# PATCH 3: Verify SDPA is being used (already present, just verify)
# The code already uses F.scaled_dot_product_attention which is good
# =============================================================================
echo "[PATCH 3] Verifying SDPA usage..."
if grep -q "scaled_dot_product_attention" "$MODULES_FILE" 2>/dev/null; then
    echo "  ✓ SDPA already in use (F.scaled_dot_product_attention)"
else
    echo "  ⚠ SDPA not found - manual attention implementation"
fi

echo "========================================"
echo "GLM-TTS patching complete!"
echo "========================================"
