# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Development Workflow

**IMPORTANT**: When receiving any command or task request:

1. **Always start from main branch**: Switch to main branch before starting any work
2. **Create new branch**: Always create a new branch with descriptive name (preferably with `claudecode/` prefix)
3. **Work in feature branch**: Perform all tasks and changes in the feature branch
4. **Create PR when complete**: Once work is finished, create a pull request
5. **Return to main**: After PR creation, always switch back to main branch

### Workflow Commands
```bash
# 1. Switch to main branch
git checkout main

# 2. Create new feature branch 
git checkout -b claudecode/feature-description

# 3. Do your work...
# (make changes, commits, etc.)

# 4. Push and create PR
git push -u origin claudecode/feature-description
gh pr create --title "Title" --body "Description"

# 5. Return to main branch
git checkout main
```

## Project Overview

Fish Speech is a multilingual text-to-speech (TTS) system with voice cloning capabilities, built with PyTorch Lightning and Hydra for configuration management. The project includes both the core TTS engine and VoiceReel, a multi-speaker TTS API service.

## Development Commands

### Environment Setup
```bash
# Install in development mode
pip install -e .

# Install with stable PyTorch dependencies
pip install -e ".[stable]"
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_server.py
python -m pytest tests/test_voicereel_client.py
python -m pytest tests/test_voicereel_infra.py
python -m pytest tests/test_docs.py

# Run with verbose output
python -m pytest -v tests/
```

### Code Quality
```bash
# Install and run pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files

# Format code with Black
black .

# Sort imports
isort --profile=black .
```

### Training and Model Operations
```bash
# Train models using Hydra configs
python fish_speech/train.py

# Extract VQ features
python tools/vqgan/extract_vq.py

# Run inference WebUI
python tools/run_webui.py

# Start API server
python tools/api_server.py
```

### VoiceReel API Development
```bash
# Start VoiceReel server
python -m voicereel.server

# Run Flask app
python -m voicereel.flask_app

# Test client functionality
python -m voicereel.client
```

### Documentation
```bash
# Serve docs locally
mkdocs serve

# Build and deploy docs
mkdocs gh-deploy --force
```

## Architecture Overview

### Core Components

**fish_speech/**: Main TTS engine package
- `models/`: Neural network architectures (text2semantic LLaMA, VQGAN)
- `datasets/`: Data loading and preprocessing for semantic tokens and VQGAN
- `train.py`: PyTorch Lightning training script with Hydra configuration
- `configs/`: Hydra configuration files for different model types and training scenarios

**tools/**: Utilities and applications
- `api_server.py`: REST API server for TTS inference
- `run_webui.py`: Gradio-based web interface
- `llama/`: LLaMA model utilities (quantization, LoRA merging, dataset building)
- `vqgan/`: VQGAN preprocessing tools

**voicereel/**: Multi-speaker TTS API service
- Production-ready B2B API for voice generation with multiple speakers
- Task queue system for async processing
- Database models for speaker management
- See `voicereel/PRD.md` for detailed API specifications

### Training Pipeline

1. **VQGAN Training**: Vector quantized audio representation learning
2. **Text2Semantic**: LLaMA-based model for text to semantic token generation
3. **LoRA Fine-tuning**: Efficient adaptation using low-rank adaptation

### Configuration System

Uses Hydra for configuration management:
- `configs/base.yaml`: Base configuration
- `configs/firefly_gan_vq.yaml`: VQGAN configuration
- `configs/text2semantic_finetune.yaml`: Fine-tuning configuration
- `configs/lora/`: LoRA-specific configurations

## Key Dependencies

- **PyTorch Lightning**: Training framework
- **Hydra**: Configuration management
- **Transformers**: HuggingFace model implementations
- **Gradio**: Web UI framework
- **Vector Quantize PyTorch**: Vector quantization implementation

## Model Management

Models are managed through:
- `tools/download_models.py`: Download pre-trained models
- `tools/extract_model.py`: Extract model weights
- `tools/export_onnx.py`: ONNX export for deployment

## Testing Infrastructure

Test files cover:
- Server functionality (`test_server.py`)
- VoiceReel client and infrastructure (`test_voicereel_*.py`)
- Documentation validation (`test_docs.py`)

## Docker Development

```bash
# Development container
docker-compose -f docker-compose.dev.yml up

# Build production image
docker build -f dockerfile .
```
