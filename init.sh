#!/bin/bash
set -e

# =============================================================================
# Fish Speech H100 本番環境最適化スクリプト
# 100同時リクエスト対応 - 80GB VRAM + 20vCPU + 251GB RAM
# 目標: 150-300 requests/second, 1:15 real-time factor
# =============================================================================


# =============================================================================
# 必要パッケージインストール
# =============================================================================

echo "📦 パッケージインストール..."

apt update && apt upgrade -y
apt install -y \
    build-essential cmake ca-certificates \
    libsox-dev libasound-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 \
    ffmpeg git wget curl

# =============================================================================
# CUDA & PyTorch環境設定（H100最適化）
# =============================================================================
echo "⚡ CUDA環境設定（H100特化）..."

# 基本CUDA設定
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# H100 80GB VRAM最適化メモリ管理
export PYTORCH_CUDA_ALLOC_CONF="backend:native,max_split_size_mb:512,roundup_power2_divisions:4,garbage_collection_threshold:0.8,expandable_segments:True"

# CUDA接続数最大化（H100対応）
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_MODULE_LOADING=LAZY
export CUDA_LAUNCH_BLOCKING=0

# H100 Transformer Engine最適化
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_ALLOW_TF32=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1

# PyTorch 2.x コンパイル最適化
export TORCH_COMPILE_DEBUG=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCH_COMPILE_MODE=max-autotune

# =============================================================================
# CPU & メモリ最適化（20コア + 251GB RAM）
# =============================================================================
echo "🧠 CPU・メモリ最適化（20コア対応）..."

# 20コアCPU最適化設定
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export VECLIB_MAXIMUM_THREADS=20
export TORCH_NUM_THREADS=20
export TORCH_NUM_INTEROP_THREADS=4

# NUMA & CPUアフィニティ最適化
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
export GOMP_CPU_AFFINITY="0-19"

# NCCL設定（マルチプロセス対応）
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# =============================================================================
# Fish Speech環境構築
# =============================================================================
echo "🐟 Fish Speech環境構築..."

# 依存関係インストール
pip install --upgrade pip
pip install -e .[stable]
pip install huggingface_hub

# =============================================================================
# モデルダウンロード & キャッシング
# =============================================================================
echo "📥 モデルダウンロード..."

huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# 251GB RAM活用 - モデル事前キャッシング
echo "💾 モデル事前キャッシング（251GB RAM活用）..."
python3 -c "
import torch
import sys
import os
sys.path.append('/workspace/fish-speech')
os.chdir('/workspace/fish-speech')

print('🔄 Fish Speech環境確認...')

# モデルファイル確認
model_path = './checkpoints/openaudio-s1-mini'
if os.path.exists(model_path):
    print(f'✅ モデルパス存在: {model_path}')
    files = os.listdir(model_path)
    print(f'📁 モデルファイル: {files}')
else:
    print(f'❌ モデルパス不存在: {model_path}')

# CUDA環境確認
if torch.cuda.is_available():
    print(f'✅ CUDA利用可能: {torch.cuda.get_device_name(0)}')
    print(f'📊 VRAM容量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
    
    # H100最適化確認
    if 'H100' in torch.cuda.get_device_name(0):
        print('🚀 H100検出 - 最適化適用')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('✅ TF32有効化完了')
    
    # メモリ確保テスト（軽量）
    test_tensor = torch.randn(1000, 1000, device='cuda')
    print(f'📊 メモリテスト成功: {torch.cuda.memory_allocated()/1024**2:.1f}MB使用')
    del test_tensor
    torch.cuda.empty_cache()
else:
    print('❌ CUDA利用不可')

print('✅ モデル事前確認完了')
"