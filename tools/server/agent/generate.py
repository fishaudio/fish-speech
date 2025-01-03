import time

from fish_speech.utils.schema import ServeMessage, ServeResponse, ServeStreamResponse
from tools.server.agent.generation_utils import (
    initialize_decode_buffers,
    process_response_tokens,
    send_reset_buffer,
)
from tools.server.agent.pre_generation_utils import (
    create_generation_request,
    send_generation_request,
)


def generate_responses(
    input_queue, tokenizer, config, request, prompt, im_end_id, device
):
    """
    Main generation function that handles the conversation, encodes the request,
    sends the generation request, and handles decoding/streaming.
    It returns a response generator (ServeResponse or ServeStreamResponse).
    """
    stats = {}
    start = time.time()
    stats["start_time"] = start
    stats["tokens_count"] = 0

    # Prepare and send the generation request
    req = create_generation_request(prompt, request, im_end_id, device)
    response_queue = send_generation_request(input_queue, req)
    decode_buffer, parts, finished = initialize_decode_buffers(request.num_samples)

    while True:
        response = response_queue.get()

        # Handle abnormal finish or error
        if response in ["stop", "error"]:
            finish_reason = response
            break

        # Process the response tokens
        is_first_token = stats["tokens_count"] == 0
        responses = process_response_tokens(
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
        )

        # Yield the responses if streaming
        if request.streaming and responses:
            for r in responses:
                yield r

        stats["tokens_count"] += 1

        # Check if all samples are finished
        if all(finished):
            finish_reason = "stop"
            break

    # Finalize the response
    final_responses = finalize_response(
        request, finished, decode_buffer, tokenizer, parts, stats, finish_reason
    )
    for fr in final_responses:
        yield fr


def finalize_response(
    request, finished, decode_buffer, tokenizer, parts, stats, finish_reason
):
    """
    Finalize the response by sending the remaining text buffers.
    """
    responses = []

    # Send the remaining text buffers
    for sample_id in range(request.num_samples):
        responses.extend(
            send_reset_buffer(sample_id, decode_buffer, tokenizer, parts, request)
        )

    # Calculate the final stats
    stats["total_time"] = (time.time() - stats["start_time"]) * 1000
    stats["total_tokens"] = stats["tokens_count"]

    # If streaming, send the final chunks for each sample
    if request.streaming:
        for sample_id in range(request.num_samples):
            if finished[sample_id]:
                continue
            responses.append(
                ServeStreamResponse(
                    finish_reason=finish_reason, stats=stats, sample_id=sample_id
                )
            )
    else:
        # If not streaming, send the full messages for each sample
        full_messages = [
            ServeMessage(role="assistant", parts=parts[i])
            for i in range(request.num_samples)
        ]
        responses.append(
            ServeResponse(
                messages=full_messages,
                finish_reason=finish_reason,
                stats=stats,
            )
        )

    return responses
