#!/bin/bash
set -e

echo "🚀 Fish Speech H100本番環境起動中..."

# NUMA最適化で起動（利用可能な場合）
if command -v numactl >/dev/null 2>&1; then
    echo "🧠 NUMA最適化で起動..."
    numactl --cpunodebind=0 --membind=0 supervisord -c /etc/supervisor/supervisord.conf
else
    echo "⚠️ numactl利用不可 - 通常起動..."
    supervisord -c /etc/supervisor/supervisord.conf
fi

# サービス状態確認
sleep 5

echo "=================== 起動完了 ==================="
echo "🐟 Fish Speech API: http://localhost:8000"
echo "📊 Health Check: http://localhost:8000/health"
echo "📈 Metrics: http://localhost:8000/metrics"
echo "🔴 Redis: localhost:6379"
echo "==============================================="

# リアルタイム監視
echo "📊 システム監視開始（Ctrl+Cで終了）..."
watch -n 1 '
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits
echo ""
echo "=== System Load ==="
uptime
echo ""
echo "=== Memory Usage ==="
free -h
echo ""
echo "=== API Status ==="
curl -s http://localhost:8000/health | jq .