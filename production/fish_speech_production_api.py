import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from concurrent.futures import ThreadPoolExecutor
import redis
import json
import time
import os
import subprocess
from typing import Optional
import logging
import tempfile
import base64

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# H100最適化設定
os.environ.update({
    'TORCH_COMPILE_DEBUG': '0',
    'CUDA_LAUNCH_BLOCKING': '0',
    'PYTORCH_CUDA_ALLOC_CONF': 'backend:native,max_split_size_mb:512,garbage_collection_threshold:0.8,expandable_segments:True'
})

# メトリクス定義
REQUEST_COUNT = Counter('fish_speech_requests_total', 'Total requests', ['status'])
REQUEST_LATENCY = Histogram('fish_speech_request_duration_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('fish_speech_active_requests', 'Active requests')
GPU_UTILIZATION = Gauge('fish_speech_gpu_utilization_percent', 'GPU utilization')
GPU_MEMORY = Gauge('fish_speech_gpu_memory_used_gb', 'GPU memory used GB')
QUEUE_SIZE = Gauge('fish_speech_queue_size', 'Queue size')

# アプリケーション設定
app = FastAPI(title="Fish Speech H100 Production API")
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# グローバル変数
executor = ThreadPoolExecutor(max_workers=22)  # 20コア + 2予備
fish_speech_path = '/workspace/fish-speech'

class AudioRequest(BaseModel):
    text: str
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    compile: bool = True

class AudioResponse(BaseModel):
    audio_data: str  # base64 encoded
    processing_time: float
    tokens_generated: int
    status: str

def run_fish_speech_inference(text: str, **kwargs):
    """Fish Speech CLIベース推論"""
    try:
        start_time = time.time()
        
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            text_file = f.name
        
        # Fish Speech推論コマンド構築
        cmd = [
            'python', 
            f'{fish_speech_path}/fish_speech/models/text2semantic/inference.py',
            '--text', text,
            '--checkpoint-path', f'{fish_speech_path}/checkpoints/openaudio-s1-mini',
            '--num-samples', '1',
            '--max-new-tokens', str(kwargs.get('max_new_tokens', 1024)),
            '--temperature', str(kwargs.get('temperature', 0.7)),
            '--top-p', str(kwargs.get('top_p', 0.9))
        ]
        
        # コンパイル最適化
        if kwargs.get('compile', True):
            cmd.append('--compile')
        
        # H100最適化
        if torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0):
            cmd.extend(['--half'])  # BF16使用
        
        # 推論実行
        os.chdir(fish_speech_path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒タイムアウト
        )
        
        if result.returncode != 0:
            logger.error(f"Fish Speech推論エラー: {result.stderr}")
            raise Exception(f"推論失敗: {result.stderr}")
        
        # 生成されたコードファイル確認
        codes_file = None
        for i in range(10):  # codes_0.npy ~ codes_9.npy
            candidate = f'codes_{i}.npy'
            if os.path.exists(candidate):
                codes_file = candidate
                break
        
        if not codes_file:
            raise Exception("セマンティックトークン生成に失敗")
        
        # 音声生成（VQGAN）
        audio_cmd = [
            'python',
            f'{fish_speech_path}/fish_speech/models/vqgan/inference.py',
            '-i', codes_file,
            '--checkpoint-path', f'{fish_speech_path}/checkpoints/openaudio-s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth'
        ]
        
        audio_result = subprocess.run(
            audio_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if audio_result.returncode != 0:
            logger.error(f"音声生成エラー: {audio_result.stderr}")
            raise Exception(f"音声生成失敗: {audio_result.stderr}")
        
        # 生成された音声ファイル読み込み
        audio_file = 'fake.wav'
        if os.path.exists(audio_file):
            with open(audio_file, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            raise Exception("音声ファイル生成に失敗")
        
        processing_time = time.time() - start_time
        
        # 一時ファイル削除
        for temp_file in [text_file, codes_file, audio_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return {
            'audio_data': audio_data,
            'processing_time': processing_time,
            'tokens_generated': len(text) // 4,  # 推定
            'status': 'success'
        }
        
    except subprocess.TimeoutExpired:
        logger.error("推論タイムアウト")
        raise HTTPException(status_code=504, detail="推論タイムアウト")
    except Exception as e:
        logger.error(f"推論エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時初期化"""
    logger.info("🚀 Fish Speech H100 API起動中...")
    
    # 環境確認
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA利用可能: {torch.cuda.get_device_name(0)}")
        logger.info(f"📊 VRAM容量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    else:
        logger.warning("⚠️ CUDA利用不可")
    
    # Fish Speechパス確認
    if os.path.exists(fish_speech_path):
        logger.info(f"✅ Fish Speechパス確認: {fish_speech_path}")
    else:
        logger.error(f"❌ Fish Speechパス不存在: {fish_speech_path}")
    
    # バックグラウンドメトリクス更新
    asyncio.create_task(update_metrics())

async def update_metrics():
    """システムメトリクス更新"""
    while True:
        try:
            # GPU情報更新
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                GPU_UTILIZATION.set(gpu.load * 100)
                GPU_MEMORY.set(gpu.memoryUsed / 1024)  # GB変換
            
            # キューサイズ更新
            queue_size = redis_client.llen('audio_queue') if redis_client.ping() else 0
            QUEUE_SIZE.set(queue_size)
            
        except Exception as e:
            logger.error(f"メトリクス更新エラー: {e}")
        
        await asyncio.sleep(1)

@app.post("/generate", response_model=AudioResponse)
async def generate_audio(request: AudioRequest):
    """音声生成エンドポイント"""
    ACTIVE_REQUESTS.inc()
    
    try:
        with REQUEST_LATENCY.time():
            # 非同期実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                run_fish_speech_inference,
                request.text,
                **request.dict(exclude={'text'})
            )
            
        REQUEST_COUNT.labels(status='success').inc()
        return AudioResponse(**result)
        
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        logger.error(f"音声生成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/metrics")
async def metrics():
    """Prometheusメトリクス"""
    return generate_latest()

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    try:
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        queue_size = redis_client.llen('audio_queue') if redis_client.ping() else 0
        
        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_gb": round(gpu_memory, 2),
            "active_requests": ACTIVE_REQUESTS._value._value,
            "queue_size": queue_size,
            "fish_speech_path": fish_speech_path,
            "fish_speech_exists": os.path.exists(fish_speech_path)
        }
    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    # H100最適化Uvicorn設定
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # シングルワーカー（GPU共有回避）
        loop="uvloop",
        http="httptools",
        access_log=False,  # パフォーマンス重視
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )