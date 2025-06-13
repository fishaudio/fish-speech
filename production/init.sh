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
# 基本パッケージインストール（最小構成）
echo "📦 基本パッケージインストール..."
apt install -y \
    build-essential cmake ca-certificates \
    libsox-dev libasound-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 \
    ffmpeg git wget curl

# 本番環境パッケージ（推奨）
echo "🚀 本番環境パッケージインストール..."
apt install -y \
    htop \
    redis-server \
    supervisor \
    util-linux

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

python3 /workspace/fish-speech/production/model_check.py

# =============================================================================
# Redis設定（キューイング用）
# =============================================================================

echo "🔴 Redis設定（高性能キューイング）..."

mkdir -p /etc/redis /var/lib/redis /var/log/redis
chown redis:redis /var/lib/redis /var/log/redis

cat > /etc/redis/redis.conf << EOF
# 基本設定
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
daemonize yes

# 高性能設定
maxmemory 8gb
maxmemory-policy allkeys-lru
save ""
stop-writes-on-bgsave-error no

# AOF無効化（パフォーマンス重視）
appendonly no

# ネットワーク最適化
tcp-backlog 2048
timeout 300

# ログ設定
loglevel notice
logfile /var/log/redis/redis-server.log
EOF


echo "👥 Supervisor設定..."

mkdir -p /etc/supervisor/conf.d

cat > /etc/supervisor/conf.d/fish-speech-production.conf << EOF
[program:redis-server]
command=/usr/bin/redis-server /etc/redis/redis.conf
autostart=true
autorestart=true
user=redis
redirect_stderr=true
stdout_logfile=/var/log/supervisor/redis.log

[program:fish-speech-api]
command=/usr/bin/python /workspace/fish-speech/production/fish_speech_production_api.py
directory=/workspace/fish-speech
autostart=true
autorestart=true
user=root
redirect_stderr=true
stdout_logfile=/var/log/supervisor/fish-speech-api.log
environment=CUDA_VISIBLE_DEVICES=0,PYTHONPATH="/workspace/fish-speech"
numprocs=1
process_name=%(program_name)s_%(process_num)02d

[group:fish-speech]
programs=redis-server,fish-speech-api
priority=999
EOF

# =============================================================================
# 起動スクリプト & モニタリング
# =============================================================================

pip install \
    huggingface_hub \
    nvidia-ml-py3 \
    uvicorn[standard] \
    fastapi \
    redis \
    celery \
    gunicorn \
    prometheus-client \
    psutil \
    gpustat

echo "🎯 起動スクリプト作成..."

chmod +x /workspace/start_fish_speech_production.sh
./production/start_fish_speech_production.sh