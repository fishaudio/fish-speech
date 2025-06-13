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
apt update && apt upgrade -y
apt install -y \
    build-essential cmake ca-certificates \
    libsox-dev libasound-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 \
    ffmpeg git wget curl

echo "🔧 本番環境パッケージインストール..."
apt install -y supervisor vim

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

cp production/fish-speech-supervisor.conf /etc/supervisor/conf.d/fish-speech.conf
mkdir -p /var/log/fish-speech
chown root:root /var/log/fish-speech
