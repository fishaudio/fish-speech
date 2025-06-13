#!/usr/bin/env python3
import torch
import sys
import os

def main():
    print('🔄 Fish Speech環境確認...')
    
    # パス設定
    fish_speech_path = '/workspace/fish-speech'
    sys.path.append(fish_speech_path)
    os.chdir(fish_speech_path)
    
    # モデルファイル確認
    model_path = './checkpoints/openaudio-s1-mini'
    if os.path.exists(model_path):
        print(f'✅ モデルパス存在: {model_path}')
        try:
            files = os.listdir(model_path)
            print(f'📁 モデルファイル: {files}')
            
            # 重要ファイルの存在確認
            important_files = ['config.yaml', 'model.pth', 'codec.pth']
            for file in important_files:
                if file in files:
                    file_size = os.path.getsize(os.path.join(model_path, file)) / 1024**2
                    print(f'   ✅ {file}: {file_size:.1f}MB')
                else:
                    print(f'   ⚠️ {file}: 不存在')
        except Exception as e:
            print(f'❌ ファイル一覧取得エラー: {e}')
    else:
        print(f'❌ モデルパス不存在: {model_path}')
        print('💡 HuggingFaceからのダウンロードが必要です')
        return False
    
    # CUDA環境確認
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f'✅ CUDA利用可能: {device_name}')
        
        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / 1024**3
        print(f'📊 VRAM容量: {vram_gb:.1f}GB')
        print(f'📊 SM数: {device_props.multi_processor_count}')
        print(f'📊 CUDAバージョン: {torch.version.cuda}')
        
        # H100最適化確認
        if 'H100' in device_name:
            print('🚀 H100検出 - 最適化適用中...')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            print('✅ TF32 + FlashAttention有効化完了')
            
            # H100固有機能確認
            if hasattr(torch.cuda, 'get_device_capability'):
                capability = torch.cuda.get_device_capability(0)
                print(f'📊 Compute Capability: {capability[0]}.{capability[1]}')
        
        # メモリテスト
        try:
            print('🧪 GPUメモリテスト実行中...')
            initial_memory = torch.cuda.memory_allocated()
            
            # 軽量テスト
            test_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
            memory_used = torch.cuda.memory_allocated() - initial_memory
            print(f'📊 メモリテスト成功: {memory_used/1024**2:.1f}MB使用')
            
            # 大容量テスト（H100の場合）
            if 'H100' in device_name and vram_gb > 50:
                print('🚀 H100大容量メモリテスト...')
                large_tensor = torch.randn(10000, 10000, device='cuda', dtype=torch.bfloat16)
                large_memory = torch.cuda.memory_allocated() - initial_memory
                print(f'📊 大容量テスト成功: {large_memory/1024**3:.2f}GB使用')
                del large_tensor
            
            del test_tensor
            torch.cuda.empty_cache()
            
            # メモリ情報
            memory_info = torch.cuda.memory_get_info()
            free_memory = memory_info[0] / 1024**3
            total_memory = memory_info[1] / 1024**3
            used_memory = total_memory - free_memory
            print(f'📊 メモリ状況: {used_memory:.1f}GB使用 / {total_memory:.1f}GB総容量')
            
        except Exception as e:
            print(f'❌ GPUメモリテストエラー: {e}')
            return False
    else:
        print('❌ CUDA利用不可')
        print('💡 NVIDIA GPUとCUDAドライバーの確認が必要です')
        return False
    
    # PyTorch最適化設定確認
    print('\n🔧 PyTorch最適化設定確認:')
    print(f'   TF32 MatMul: {torch.backends.cuda.matmul.allow_tf32}')
    print(f'   TF32 cuDNN: {torch.backends.cudnn.allow_tf32}')
    print(f'   cuDNN Benchmark: {torch.backends.cudnn.benchmark}')
    print(f'   cuDNN Deterministic: {torch.backends.cudnn.deterministic}')
    
    # 環境変数確認
    print('\n🌍 重要な環境変数:')
    important_env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF', 
        'TORCH_COMPILE_DEBUG',
        'OMP_NUM_THREADS'
    ]
    
    for var in important_env_vars:
        value = os.environ.get(var, '未設定')
        print(f'   {var}: {value}')
    
    print('\n✅ モデル事前確認完了')
    return True

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print('\n🎉 環境確認成功 - Fish Speech実行準備完了')
            sys.exit(0)
        else:
            print('\n❌ 環境確認失敗 - 設定を見直してください')
            sys.exit(1)
    except KeyboardInterrupt:
        print('\n⚠️ ユーザーによる中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n💥 予期しないエラー: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)