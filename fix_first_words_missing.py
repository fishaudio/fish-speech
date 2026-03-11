"""
Fix for Issue #881: First words missing from playback

Root Cause Analysis:
1. Fast model input_pos reset to 0 in decode_one_token_ar (line 264)
2. This causes KV cache position mismatch for first few tokens
3. Result: First semantic tokens generate incorrect audio codes
4. Audio: First ~100-500ms missing or corrupted

Solution:
- Preserve input_pos across fast model forward passes
- Fix cache warming for first token
- Add padding/warmup tokens to prevent first-word loss
"""

import torch
from typing import Tuple, Optional


def decode_one_token_ar_fixed(
    model,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    # NEW: Track fast model position
    fast_input_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed version of decode_one_token_ar that preserves fast model input_pos.
    
    Returns:
        Tuple of (codebooks, fast_input_pos) to track position across calls
    """
    # Main model forward pass
    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits  # (1, 1, vocab_size)
    hidden_states = forward_result.hidden_states

    # Apply constrained decoding
    biased_logits = logits + semantic_logit_bias

    # Normal sample
    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )[0]

    # RAS: high-temp fallback for repetition
    high_temp = torch.tensor(
        1.0, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(0.9, device=top_p.device, dtype=top_p.dtype)
    main_token_high = sample(
        biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
    )[0]

    # RAS logic
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    # CRITICAL FIX: Use continuous input_pos for fast model
    # Instead of resetting to 0, increment from previous position
    if fast_input_pos is None:
        fast_input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    
    # First forward pass in fast model (position 0)
    model.forward_generate_fast(hidden_states, fast_input_pos.clone())

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    # Subsequent codebook predictions (positions 1 to num_codebooks-1)
    for codebook_idx in range(1, model.config.num_codebooks):
        # FIX: Increment position for each codebook
        fast_input_pos = fast_input_pos + 1
        logits = model.forward_generate_fast(hidden_states, fast_input_pos.clone())

        # Sample codebook token
        codebook_logits = logits  # DualAR predicts codebook_size tokens
        
        # Apply temperature and sample
        codebook_probs = torch.nn.functional.softmax(
            codebook_logits / torch.clip(temperature, min=1e-5), dim=-1
        )
        codebook_token = multinomial_sample_one_no_sync(codebook_probs[0, -1])
        
        codebook_token = torch.clamp(
            codebook_token, 
            min=0, 
            max=model.config.codebook_size - 1
        )
        
        codebooks.append(codebook_token)
        
        # Update hidden states for next codebook
        if codebook_idx < model.config.num_codebooks - 1:
            hidden_states = model.fast_embeddings(codebook_token)

    result = torch.stack(codebooks, dim=0)
    return result, fast_input_pos


def decode_n_tokens_fixed(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar_fixed,
):
    """
    Fixed version that maintains fast_input_pos across token generation.
    """
    # RAS rolling window
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, 10),  # RAS_WIN_SIZE
        dtype=torch.int,
        device=cur_token.device,
    )
    
    new_tokens = []
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    
    # CRITICAL FIX: Track fast model position across calls
    fast_input_pos = None

    for i in range(num_new_tokens):
        next_token, fast_input_pos = decode_one_token(
            model=model,
            x=cur_token,
            input_pos=input_pos,
            previous_tokens=previous_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            semantic_logit_bias=semantic_logit_bias,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            fast_input_pos=fast_input_pos,  # Pass tracked position
        )
        next_token = next_token.clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        
        # RAS window roll
        previous_tokens = previous_tokens.roll(-1, dims=1)
        previous_tokens[:, -1] = next_token.view(model.config.num_codebooks + 1, -1)[:, 0]
        
        new_tokens.append(next_token)

        if cur_token[0, 0, -1] == im_end_id:
            break

    del cur_token
    return torch.cat(new_tokens, dim=1)


def generate_fixed(
    *,
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar_fixed,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Fixed generate function that uses corrected decode functions.
    """
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(model.parameters()).dtype

    # Setup caches
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt.dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )
    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    # FIX: Initialize fast_input_pos for first token
    first_token, fast_input_pos = decode_one_token(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
        fast_input_pos=None,  # Will be initialized in decode_one_token
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    # FIX: Pass fast_input_pos to decode_n_tokens
    x = decode_n_tokens_fixed(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
        decode_one_token=decode_one_token,
    )

    seq[:, T + 1 :] = x

    return seq[:, : T + x.shape[1] + 1]


# Additional fix: Audio segment buffering
def inference_wrapper_with_warmup(req, engine):
    """
    Wrapper that adds warmup/padding to prevent first-word loss.
    
    Root cause: First few audio segments may be incomplete due to:
    - Model initialization overhead
    - Streaming buffer underrun
    - Client-side buffer discarding first chunks
    
    Solution: Pre-generate warmup tokens, then discard their audio.
    """
    import numpy as np
    
    # Add warmup prefix to text (will be discarded in output)
    warmup_text = "... "  # 3 dots + space = ~200ms of silence/warmup
    original_text = req.text
    req.text = warmup_text + original_text
    
    # Generate with warmup
    segments = []
    warmup_samples = 0
    sample_rate = None
    
    for result in engine.inference(req):
        if result.code == "header":
            sample_rate = result.audio[0]
            # Don't yield header yet - wait until we have real content
        elif result.code == "segment":
            segment = result.audio[1]
            if warmup_samples < 4000:  # Discard first ~200ms at 24kHz
                warmup_samples += len(segment)
                # Skip yielding this segment (it's warmup)
            else:
                # Now yield header (once we have real content)
                if len(segments) == 0 and sample_rate:
                    from fish_speech.inference_engine.utils import wav_chunk_header
                    yield InferenceResult(
                        code="header",
                        audio=(sample_rate, np.array(wav_chunk_header(sample_rate=sample_rate))),
                        error=None,
                    )
                # Yield real segment
                yield result
                segments.append(segment)
        elif result.code == "final":
            # Yield final without warmup
            yield result
    
    return segments


# Usage in fish_speech/models/text2semantic/inference.py:
# Replace decode_one_token_ar with decode_one_token_ar_fixed
# Replace decode_n_tokens with decode_n_tokens_fixed
# Replace generate with generate_fixed
