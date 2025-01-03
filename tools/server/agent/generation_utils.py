import time

from fish_speech.utils.schema import (
    ServeStreamDelta,
    ServeStreamResponse,
    ServeTextPart,
    ServeVQPart,
)


def initialize_decode_buffers(num_samples):
    """Initialise the decode buffers for each sample."""
    decode_buffer = [[] for _ in range(num_samples)]
    parts = [[] for _ in range(num_samples)]
    finished = [False for _ in range(num_samples)]
    return decode_buffer, parts, finished


def send_reset_buffer(sample_id, decode_buffer, tokenizer, parts, request):
    """Send the remaining text buffer for a sample."""
    if len(decode_buffer[sample_id]) == 0:
        return []

    decoded = tokenizer.decode(decode_buffer[sample_id])
    part = ServeTextPart(text=decoded)

    responses = []
    if request.streaming:
        responses.append(ServeStreamResponse(delta=ServeStreamDelta(part=part)))
    else:
        parts[sample_id].append(part)

    decode_buffer[sample_id] = []
    return responses


def handle_semantic_tokens(tokens, config, sample_id, parts, request):
    """Handle the semantic tokens returned by the model."""
    responses = []
    _tokens = tokens[1:].clone()

    if not config.share_codebook_embeddings:
        for i in range(len(_tokens)):
            _tokens[i] -= config.codebook_size * i

    # If streaming, send the VQ parts directly
    if request.streaming:
        responses.append(
            ServeStreamResponse(
                sample_id=sample_id,
                delta=ServeStreamDelta(part=ServeVQPart(codes=_tokens.tolist())),
            )
        )
    else:
        # If not streaming, accumulate the VQ parts
        if not parts[sample_id] or not isinstance(parts[sample_id][-1], ServeVQPart):
            parts[sample_id].append(ServeVQPart(codes=_tokens.tolist()))
        else:
            # Accumulate the codes
            for codebook_id, value in enumerate(_tokens):
                parts[sample_id][-1].codes[codebook_id].append(value.item())

    return responses


def process_response_tokens(
    response,
    tokenizer,
    config,
    request,
    decode_buffer,
    parts,
    finished,
    im_end_id,
    stats,
    start,
    is_first_token,
):
    """Process the response tokens returned by the model."""
    responses = []
    for sample_id, tokens in enumerate(response):
        if finished[sample_id]:
            continue

        # End of the conversation
        if tokens[0] == im_end_id:
            finished[sample_id] = True
            # Send the remaining text buffer
            responses.extend(
                send_reset_buffer(sample_id, decode_buffer, tokenizer, parts, request)
            )
            if request.streaming:
                responses.append(
                    ServeStreamResponse(
                        sample_id=sample_id,
                        finish_reason="stop",
                        stats=stats,
                    )
                )
            continue

        # Check if the token is semantic
        is_semantic = (
            tokenizer.semantic_begin_id <= tokens[0] <= tokenizer.semantic_end_id
        )

        if is_semantic:
            # Before the semantic tokens, send the remaining text buffer
            responses.extend(
                send_reset_buffer(sample_id, decode_buffer, tokenizer, parts, request)
            )
            responses.extend(
                handle_semantic_tokens(tokens, config, sample_id, parts, request)
            )
        else:
            # Accumulate the text tokens (not implemented?)
            decode_buffer[sample_id].append(tokens[0, 0])

    if is_first_token:
        stats["time_to_first_token"] = (time.time() - start) * 1000

    return responses
