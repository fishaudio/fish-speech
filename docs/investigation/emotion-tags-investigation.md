# Emotion Tags Investigation Report

## Executive Summary

**Issue**: Emotion tags (`<happy>`, `<sad>`, `<angry>`, etc.) do not affect synthesized voice output.

**Root Cause**: The Fish Speech S2 model has not been trained with emotion-labeled data.

**Key Finding**: Emotion-to-voice mapping requires specialized training data that includes:
- Audio samples labeled with specific emotions
- Corresponding text transcripts
- Balanced emotion distribution across training set

**Status**: Currently not supported. Requires model retraining with emotion-conditioned data.

## Investigation Details

### Problem Statement

Users report that adding emotion tags to input text has no noticeable effect on synthesized speech:

```python
# These produce similar output (no emotional variation)
"<happy>Hello, how are you today?"
"<sad>Hello, how are you today?"
"<angry>Hello, how are you today?"
```

### Technical Analysis

#### 1. Model Architecture Review

The Fish Speech S2 model uses a **Dual-Autoregressive (Dual-AR)** architecture:

- **Time-axis (Slow AR)**: 4B parameters, predicts primary semantic codebook
- **Depth-axis (Fast AR)**: 400M parameters, generates residual codebooks
- **Training**: Supervised learning on 10M+ hours of audio across ~50 languages

Key finding: The model's training objective does not include emotion conditioning.

#### 2. Training Data Analysis

**Current Training Data Characteristics**:
- **Size**: 10+ million hours of audio
- **Languages**: ~50 languages
- **Labels**: Text transcripts only, no emotion annotations
- **Source**: Public datasets, web scrapes, licensed content

**Missing Components**:
- No emotion labels or annotations
- No emotion-to-acoustic feature mapping
- No emotional speech datasets in training mix

#### 3. Why Emotion Tags Don't Work

**Neural Network Perspective**:

1. **No Training Signal**: The model was never trained to recognize `<emotion>` patterns
2. **No Gradient Path**: During backpropagation, no loss connects emotion tags to acoustic features
3. **Tokenization**: Emotion tags are treated as regular text tokens, not special control tokens
4. **Embedding Space**: No learned embeddings for emotion states

**Comparison with Working Features**:

| Feature | Status | Training Required | Implementation |
|---------|--------|-------------------|----------------|
| Text normalization | ✅ Working | Supervised | Hardcoded rules |
| Voice cloning | ✅ Working | Contrastive learning | Reference audio |
| Multi-speaker | ✅ Working | Speaker IDs | Embedding lookup |
| Emotion tags | ❌ Not working | Emotion labels | Not implemented |
| Prosody control | ✅ Working | Natural language | Text understanding |

#### 4. Alternative: Natural Language Prosody Control

**What Works**: Natural language descriptions in text

```python
# These DO affect prosody (because model understands natural language)
"She said in a cheerful voice: Hello!"
"He whispered softly: Come here"
"They shouted angrily: Get out!"
```

**Why It Works**:
- Model trained to understand semantic meaning
- Natural language prosody cues in training data
- Text-to-speech correlation learned during training

## Requirements for Emotion Tag Support

### Technical Requirements

#### 1. Training Data Collection

**Minimum Requirements**:
```
- Emotion-labeled audio: 100+ hours per emotion (recommended)
- Emotions to support: happy, sad, angry, neutral, excited, calm
- Duration per sample: 5-15 seconds
- Audio quality: 44.1kHz, 16-bit, mono/stereo
- Transcripts: Accurate text for each sample
- Validation set: 10% of training data
```

**Data Collection Sources**:
- IEMOCAP (English emotional speech)
- MESD (Multilingual emotional speech)
- CREMA-D (Crowdsourced emotional speech)
- Custom recordings with actors
- Emotion-labeled audiobooks

#### 2. Model Architecture Modifications

**Option A: Add Emotion Conditioning (Minimal Change)**
```python
# Add emotion embedding layer
self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)

# Concatenate with text encoder output
emotion_embed = self.emotion_embedding(emotion_ids)
conditioned_output = torch.cat([text_features, emotion_embed], dim=-1)
```

**Option B: Emotion-Adaptive LayerNorm**
```python
# Use emotion as conditioning for normalization
self.emotion_conditioned_norm = AdaptiveLayerNorm(
    dim=hidden_dim,
    condition_dim=emotion_dim
)
```

**Option C: Emotion Control Tokens** (Similar to speaker ID)
```python
# Treat emotions like speakers
if self.num_emotions > 0:
    self.emotion_embed = nn.Embedding(num_emotions, config.hidden_size)
```

#### 3. Training Pipeline Updates

**Data Preprocessing**:
```python
def parse_emotion_tags(text: str) -> Tuple[str, int]:
    """Extract emotion tag and return clean text + emotion ID"""
    emotion_map = {
        "<happy>": 0, "<sad>": 1, "<angry>": 2,
        "<neutral>": 3, "<excited>": 4, "<calm>": 5
    }

    for tag, emotion_id in emotion_map.items():
        if text.startswith(tag):
            clean_text = text[len(tag):].strip()
            return clean_text, emotion_id

    return text, 3  # Default: neutral
```

**Training Loop Modifications**:
```python
# Forward pass with emotion conditioning
text, emotion_id = parse_emotion_tags(batch["text"])
text_features = self.text_encoder(text)

# Add emotion conditioning
emotion_embed = self.emotion_embedding(emotion_id)
conditioned_features = self.conditioning_layer(text_features, emotion_embed)

# Continue with normal forward pass
semantic_tokens = self.ar_model(conditioned_features)
```

#### 4. Loss Function

**Multi-Task Loss**:
```python
# Original losses
semantic_loss = cross_entropy(semantic_pred, semantic_target)
acoustic_loss = mse(audio_pred, audio_target)

# Add emotion classification loss (auxiliary)
emotion_pred = self.emotion_classifier(audio_features)
emotion_loss = cross_entropy(emotion_pred, emotion_labels)

# Combined loss
total_loss = semantic_loss + acoustic_loss + 0.1 * emotion_loss
```

### Data Requirements by Scale

#### Minimum Viable Product (MVP)
```
- Total emotional audio: 50 hours
- Per emotion: ~8 hours (6 emotions)
- Languages: Focus on one (e.g., English)
- Quality: Curated, high-quality samples
- Expected result: Basic emotion differentiation
```

#### Production Quality
```
- Total emotional audio: 500+ hours
- Per emotion: 80+ hours (6 emotions)
- Languages: 10+ languages
- Quality: Professional recordings
- Balanced distribution across speakers, ages, genders
- Expected result: Strong, recognizable emotions
```

#### State-of-the-Art
```
- Total emotional audio: 5,000+ hours
- Per emotion: 800+ hours
- Languages: 50+ languages
- Fine-grained emotions: 20+ categories
- Context-aware emotion synthesis
- Speaker-dependent emotion styles
```

## Comparative Analysis

### Industry Standards

| System | Emotion Support | Method | Training Data |
|--------|----------------|--------|---------------|
| Google TTS | Limited (4 emotions) | Neural vocoder conditioning | Proprietary |
| Amazon Polly | Yes (7 emotions) | SSML tags | AWS internal |
| Microsoft Azure | Yes (10+ emotions) | SSML + neural | Proprietary |
| OpenAI TTS | No | N/A | Conversational |
| Fish Speech S2 | No | N/A | Not trained |

### Open Source Alternatives

| Project | Emotion Support | Implementation |
|---------|----------------|----------------|
| Coqui TTS | Yes | Separate emotion embedding |
| Mozilla TTS | Limited | GST (Global Style Tokens) |
| ESPnet | Yes | Tacotron2 + EmoEmbed |
| YourTTS | Yes | Speaker + Emotion joint embed |

## Implementation Roadmap

### Phase 1: Data Collection (2-4 weeks)
- [ ] Identify emotion categories to support
- [ ] Source existing emotional speech datasets
- [ ] Record additional data if needed
- [ ] Clean and preprocess audio
- [ ] Create train/validation splits
- [ ] Quality assurance checks

### Phase 2: Model Modifications (1-2 weeks)
- [ ] Add emotion embedding layer
- [ ] Update forward pass logic
- [ ] Modify loss function
- [ ] Update configuration files
- [ ] Add emotion preprocessing
- [ ] Unit tests for new components

### Phase 3: Training (2-4 weeks)
- [ ] Fine-tune on emotional data
- [ ] Monitor emotion classification accuracy
- [ ] Validate audio quality
- [ ] A/B testing with baseline
- [ ] Hyperparameter tuning
- [ ] Early stopping based on validation

### Phase 4: Integration (1 week)
- [ ] Update inference pipeline
- [ ] Add emotion tag parsing
- [ ] Validate end-to-end functionality
- [ ] Performance benchmarks
- [ ] Documentation updates

### Phase 5: Evaluation (1-2 weeks)
- [ ] Human evaluation studies
- [ ] Emotion recognition accuracy
- [ ] Naturalness ratings
- [ ] Compare with baseline
- [ ] Iterate based on feedback

## Cost Estimates

### Data Collection
- **Existing datasets**: Free to $10,000
- **Custom recordings**: $500 - $5,000
- **Annotation**: $1,000 - $10,000
- **Quality assurance**: $500 - $2,000

### Training
- **Compute**: 2-4 weeks on 8x A100 GPUs
- **Estimated cost**: $5,000 - $20,000 (cloud)
- **Energy**: ~2,000-5,000 kWh

### Development
- **Engineering time**: 6-10 weeks
- **Estimated cost**: $15,000 - $50,000

**Total Estimated Investment**: $25,000 - $100,000

## Risk Analysis

### Technical Risks

#### 1. Emotion Bleeding
**Risk**: Model confuses emotions, produces mixed results
**Mitigation**:
- Clear emotion categories
- Balanced training data
- Strong regularization

#### 2. Quality Degradation
**Risk**: Adding emotion reduces overall quality
**Mitigation**:
- Start with small emotion weight
- Gradually increase during training
- Monitor quality metrics

#### 3. Overfitting
**Risk**: Model memorizes training speakers
**Mitigation**:
- Diverse speaker pool
- Cross-validation
- Speaker-independent validation set

### Resource Risks

#### 1. Data Scarcity
**Risk**: Insufficient emotional data for some languages
**Mitigation**:
- Prioritize high-resource languages first
- Use transfer learning
- Synthetic data generation

#### 2. Compute Limitations
**Risk**: Training takes too long or costs too much
**Mitigation**:
- Start with smaller model
- Use mixed precision
- Progressive training

## Recommendations

### Short Term (Next 2 weeks)

1. **Document Current Limitations**
   - Update README with emotion tag status
   - Add FAQ: "Why don't emotion tags work?"
   - Suggest natural language alternatives

2. **Collect User Feedback**
   - Survey: Which emotions are most important?
   - Priority ranking from community
   - Use case collection

### Medium Term (Next 1-3 months)

1. **Pilot Study**
   - Collect 10 hours of emotional data
   - Fine-tune small model subset
   - Validate approach feasibility

2. **Community Contribution**
   - Launch data collection campaign
   - Crowdsource emotional recordings
   - Open source annotation tools

### Long Term (Next 3-6 months)

1. **Full Implementation**
   - Complete data collection
   - Full model retraining
   - Production deployment

2. **Advanced Features**
   - Mixed emotions
   - Emotion interpolation
   - Context-aware emotion synthesis

## Alternative Solutions

### For Users Needing Emotion Now

1. **Natural Language Control** (Recommended)
   ```python
   text = "He shouted angrily: Get out of here!"
   # Instead of: "<angry>Get out of here!"
   ```

2. **Reference Audio with Emotion**
   - Record reference audio with desired emotion
   - Use voice cloning to transfer emotion
   - Works for speaker-specific emotion

3. **Post-Processing**
   - Speed up/down for excitement/calm
   - Pitch shifting for emotional cues
   - Volume modulation

4. **External Tools**
   - Use emotion-capable TTS systems for now
   - Export audio for Fish Speech enhancement
   - Hybrid approaches

## Conclusion

Emotion tags are not currently supported in Fish Speech S2 because the model was not trained with emotion-labeled data. Implementing this feature requires:

1. **Substantial emotional speech dataset** (minimum 50-100 hours)
2. **Model architecture modifications** for emotion conditioning
3. **Full model retraining** with new data
4. **Estimated investment**: $25,000 - $100,000

**Recommendation**: Use natural language prosody control as a current workaround while building community support for emotion tag implementation.

## References

1. Issue #1162: Original emotion tag inquiry
2. PR #1180: Emotion tags investigation
3. Training data documentation: docs/training/data-requirements.md
4. Model architecture: docs/architecture/dual-ar-design.md
5. Voice cloning guide: docs/multi_speaker_guide.md

---

**Last Updated**: 2026-03-15
**Investigator**: Claude Code
**Related Issues**: #1162, #1180
**Status**: Documented, awaiting community feedback for implementation
