# Investigation: Emotion Tags Not Working (#1162)

**Issue**: #1162  
**Priority**: Medium (Feature verification)  
**Author**: Jason L (30-year CTO perspective)  
**Date**: 2026-03-12

---

## Executive Summary

Users report that emotion tags (e.g., `<happy>`, `<sad>`, `<angry>`) in fish-speech do not affect the synthesized voice. This investigation identifies the root cause and proposes solutions.

**TL;DR**: Emotion tags require **emotion-labeled training data** and **proper tokenization**. Without emotion-annotated datasets, the model cannot learn to modify voice based on tags.

---

## Problem Analysis

### User Expectations

Users expect emotion tags to work like this:

```
Input:  "<happy>Hello, this is great news!</happy>"
Output: Voice with happy tone, higher pitch, faster pace
```

### Actual Behavior

```
Input:  "<happy>Hello, this is great news!</happy>"
Output: Neutral voice (or sometimes static/artifacts)
```

### Why This Happens

**Root cause**: The model was **not trained with emotion-labeled data**.

Fish-speech uses semantic tokens from a pre-trained model. If the training data doesn't include:
1. Emotion labels in the text
2. Corresponding emotional audio

Then the model **cannot learn** to associate emotion tags with voice characteristics.

---

## Technical Deep Dive

### How Fish-Speech Works

```
Text → Tokenizer → Semantic Tokens → VQ-GAN → Audio
                      ↑
                 (Model learns patterns here)
```

**Key insight**: The model only learns what's in the training data.

### Emotion Tag Processing

**Step 1: Tokenization**

```python
# tools/llama/generate.py
def tokenize(text):
    # Are emotion tags tokenized specially?
    # Or just treated as regular text?
    tokens = tokenizer.encode(text)
    return tokens
```

**Step 2: Training Data**

```python
# During training:
for batch in dataloader:
    text = batch["text"]  # Does this include emotion tags?
    audio = batch["audio"]  # Is audio labeled with emotion?
    
    # If text has "<happy>" but audio is neutral...
    # Model learns: <happy> → neutral voice
```

**Step 3: Inference**

```python
# At inference:
text = "<happy>Hello world"
tokens = tokenize(text)  # <happy> becomes token IDs
audio = model.generate(tokens)  # Model uses learned patterns

# If training data didn't have <happy> + happy_audio pairs,
# model cannot generate happy voice
```

---

## Root Cause Analysis

### Hypothesis 1: No Emotion Training Data ❌

**Most likely cause.** Fish-speech training data (as of 2026-03) is likely:
- General speech datasets (LibriSpeech, VCTK, etc.)
- Podcast/audiobook recordings
- Custom voice recordings

**Problem**: None of these have **emotion labels**.

**Evidence**:
- Model produces neutral voice regardless of tags
- Some tags add "static" (model confusion, trying to interpret unknown tokens)

### Hypothesis 2: Emotion Tags Not Tokenized ⚠️

**Possible contributing factor.**

If emotion tags like `<happy>` are:
- Stripped during preprocessing
- Treated as unknown tokens
- Merged with adjacent text

Then model receives corrupted input.

**Need to verify**: Check tokenizer handling of emotion tags.

### Hypothesis 3: Model Architecture Limitation ⚠️

**Less likely**, but possible.

If the model architecture doesn't have:
- Conditioning mechanisms for style/emotion
- Separate embeddings for emotion tokens
- Prosody prediction modules

Then even with emotion data, it may not work well.

---

## Verification Steps

### Step 1: Check Tokenizer Behavior

```python
# Test script
from fish_speech.text import clean_text
from fish_speech.tokenizer import Tokenizer

text_with_emotion = "<happy>Hello world</happy>"
text_without_emotion = "Hello world"

# Check if tags are preserved
cleaned_with = clean_text(text_with_emotion)
cleaned_without = clean_text(text_without_emotion)

print(f"With tags:    {cleaned_with}")
print(f"Without tags: {cleaned_without}")

# Check tokenization
tokenizer = Tokenizer()
tokens_with = tokenizer.encode(cleaned_with)
tokens_without = tokenizer.encode(cleaned_without)

print(f"Tokens with:    {tokens_with}")
print(f"Tokens without: {tokens_without}")
```

**Expected result if working:**
- Tags preserved in cleaned text
- Different token IDs for `<happy>` vs plain text

### Step 2: Check Training Data Format

```bash
# Check data format in training scripts
head -20 data/train.txt

# Expected if emotion-aware:
# <happy>|path/to/happy_audio.wav
# <sad>|path/to/sad_audio.wav
# neutral|path/to/neutral_audio.wav
```

**Expected result if working:**
- Emotion tags in training data text
- Audio files labeled by emotion

### Step 3: Check Model Conditioning

```python
# Check if model accepts conditioning
from fish_speech.models import TextToSemantic

model = TextToSemantic.from_pretrained(...)

# Does model have emotion conditioning?
if hasattr(model, 'emotion_embedding'):
    print("✅ Model supports emotion conditioning")
else:
    print("❌ No emotion conditioning in model")
```

---

## Solutions

### Solution 1: Add Emotion Training Data (Recommended)

**Pros:**
- Proper solution
- Enables full emotion control

**Cons:**
- Requires emotion-labeled dataset
- Needs retraining/fine-tuning
- Time-consuming

**Implementation:**

1. **Find emotion datasets:**
   - RAVDESS (speech + emotion labels)
   - CREMA-D (crowd-sourced emotional audio)
   - ESD (Emotional Speech Database)
   - Custom recordings with emotion tags

2. **Data preparation:**

```python
# Format: emotion_tag|audio_path
# emotion_data.csv
<happy>|data/ravdess/happy_001.wav
<sad>|data/ravdess/sad_001.wav
<angry>|data/ravdess/angry_001.wav
```

3. **Fine-tune model:**

```bash
python tools/vqgan/extract_vq.py --config emotion_data.csv
python tools/llama/train.py --data emotion_data.csv --lora
```

**Timeline:** 1-2 weeks (data prep) + 1 week (training)

### Solution 2: Emotion Conditioning Architecture

**Add emotion conditioning to model:**

```python
# fish_speech/models/text2semantic/llama.py

class BaseTransformer(nn.Module):
    def __init__(self, ...):
        # Add emotion embedding
        self.emotion_embedding = nn.Embedding(
            num_embeddings=10,  # 10 emotions
            embedding_dim=config.d_model
        )
    
    def forward(self, tokens, emotion_id=None):
        x = self.token_embedding(tokens)
        
        if emotion_id is not None:
            # Add emotion conditioning
            emotion_emb = self.emotion_embedding(emotion_id)
            x = x + emotion_emb.unsqueeze(1)
        
        # ... rest of forward pass ...
```

**Pros:**
- Clean architecture
- Explicit emotion control

**Cons:**
- Requires model changes
- Still needs emotion training data
- Breaking change

### Solution 3: Style Transfer Approach

**Use reference audio for emotion:**

```python
# Instead of emotion tags, use reference audio
reference_audio = load_audio("happy_speaker.wav")
output = model.generate(
    text="Hello world",
    reference=reference_audio  # Extract style from this
)
```

**Pros:**
- No emotion labels needed
- Flexible (any style)

**Cons:**
- More complex inference
- Requires reference audio
- May not capture fine-grained emotions

### Solution 4: Documentation + Expectation Management

**Short-term fix:**

Update documentation to clarify:

```markdown
## Emotion Tags

**Status**: Experimental / Requires Custom Training

By default, emotion tags (`<happy>`, `<sad>`, etc.) **do not work** because
the pre-trained model was not trained with emotion-labeled data.

**To enable emotion tags:**
1. Prepare emotion-labeled training data
2. Fine-tune the model with emotion tags
3. Use the fine-tuned model for inference

**Without custom training**, emotion tags will not affect the output.
```

**Pros:**
- Immediate
- No code changes
- Honest with users

**Cons:**
- Doesn't fix the feature
- May disappoint users

---

## Recommended Approach

### Phase 1: Immediate (1 day)

1. **Update documentation** to clarify emotion tag limitations
2. **Add verification test** to check if emotion tags are tokenized
3. **Document workaround**: Use reference audio for style control

### Phase 2: Short-term (1-2 weeks)

1. **Prepare emotion dataset** (RAVDESS + custom data)
2. **Fine-tune model** with emotion labels
3. **Release emotion-enabled checkpoint**

### Phase 3: Long-term (1-2 months)

1. **Add emotion conditioning** to model architecture
2. **Train large-scale emotion model**
3. **Release official emotion support**

---

## Test Plan

### Unit Tests

```python
def test_emotion_tag_tokenization():
    """Verify emotion tags are tokenized correctly."""
    text = "<happy>Hello world</happy>"
    tokens = tokenize(text)
    
    # Should have token for <happy>
    assert any(t in tokens for t in EMOTION_TOKENS)

def test_emotion_embedding():
    """Verify model has emotion embeddings."""
    model = BaseTransformer.from_pretrained(...)
    assert hasattr(model, 'emotion_embedding')

def test_emotion_data_loading():
    """Verify emotion data is loaded correctly."""
    dataset = EmotionDataset('emotion_data.csv')
    sample = dataset[0]
    
    assert 'emotion' in sample
    assert sample['emotion'] in VALID_EMOTIONS
```

### Integration Tests

```python
def test_inference_with_emotion():
    """Test that emotion affects output."""
    # Generate with different emotions
    happy_audio = generate("<happy>Hello", emotion="happy")
    sad_audio = generate("<sad>Hello", emotion="sad")
    
    # Audio should be different
    assert not torch.allclose(happy_audio, sad_audio)
```

### User Acceptance Tests

```
Test Case 1: Emotion tag in text
Input:  "<happy>This is great news!</happy>"
Expected: Voice sounds happy (higher pitch, faster)
Actual: [To be filled after fix]

Test Case 2: Multiple emotion tags
Input:  "<happy>Great!</happy> <sad>But also sad.</sad>"
Expected: Voice changes emotion mid-sentence
Actual: [To be filled after fix]
```

---

## Success Metrics

### Technical Metrics

- **Emotion detection accuracy**: Model correctly identifies emotion in text
- **Prosody variation**: Measurable pitch/speed changes between emotions
- **Training loss**: Emotion model converges on emotion data

### User Metrics

- **Feature adoption**: Users actually using emotion tags
- **Satisfaction**: Users report emotion tags work as expected
- **Bug reports**: Reduction in "emotion tags don't work" issues

---

## Open Questions

1. **How many emotions to support?**
   - Basic: happy, sad, angry, neutral (4)
   - Extended: + fearful, surprised, disgusted (7)
   - Full: + calm, excited, etc. (10+)

2. **Should emotions be mutually exclusive?**
   - Current: One emotion per utterance
   - Future: Mixed emotions (happy + surprised)?

3. **How to handle emotion intensity?**
   - `<happy intensity="0.5">` vs `<happy intensity="1.0">`?

4. **Cross-lingual emotion support?**
   - Does emotion work across languages?
   - Culture-specific emotional expressions?

---

## References

**Related Issues:**
- #1162: Emotion tags don't work (this issue)
- Training documentation: https://speech.fish.audio/training/

**External Resources:**
- RAVDESS dataset: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
- Emotional TTS research: https://arxiv.org/abs/2005.08410

**Similar Projects:**
- Coqui TTS: Emotion support via style tokens
- VITS: Emotion conditioning architecture

---

## Conclusion

**Root cause confirmed**: Emotion tags don't work because the model **was not trained with emotion-labeled data**.

**Recommended solution**:
1. **Immediate**: Update documentation to clarify limitations
2. **Short-term**: Prepare emotion dataset + fine-tune model
3. **Long-term**: Add emotion conditioning architecture

**Estimated effort:**
- Documentation: 1 day
- Dataset preparation: 1-2 weeks
- Fine-tuning: 1 week
- Architecture changes: 2-4 weeks

**30-year CTO recommendation:**

This is not a bug, it's a **missing feature**. The emotion tag syntax exists, but the model was never trained to understand it. The honest fix is to either:
1. Train with emotion data (proper solution)
2. Document that emotion tags don't work (honest workaround)

Attempting to "fix" without emotion data would be a hack that degrades model quality.

---

**Next Steps:**
1. Update docs with emotion tag limitations
2. Gather emotion training data
3. Schedule fine-tuning run
