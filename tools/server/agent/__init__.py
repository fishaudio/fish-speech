import struct
from functools import partial

import ormsgpack

from tools.server.agent.generate import generate_responses
from tools.server.agent.pre_generation_utils import prepare_messages


def execute_request(input_queue, tokenizer, config, request, device):
    """
    This function prepares the conversation, encodes the request,
    sends the generation request, and handles decoding/streaming.
    It returns a response generator (ServeResponse or ServeStreamResponse).
    """
    prompt, im_end_id = prepare_messages(request, tokenizer, config)
    yield from generate_responses(
        input_queue, tokenizer, config, request, prompt, im_end_id, device
    )


def response_generator(req, llama_queue, tokenizer, config, device):
    """
    Non-streaming response wrapper for the chat endpoint.
    Only returns the final result.
    """
    generator = execute_request(llama_queue, tokenizer, config, req, device)
    return next(generator)


async def streaming_generator(req, llama_queue, tokenizer, config, device, json_mode):
    """
    Streaming response wrapper for the chat endpoint.
    Returns the response in chunks.
    """
    generator = execute_request(llama_queue, tokenizer, config, req, device)
    for i in generator:
        if json_mode:
            body = i.model_dump_json().encode("utf-8")
            yield b"data: " + body + b"\n\n"
        else:
            body = ormsgpack.packb(i, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
            yield struct.pack("I", len(body)) + body


def get_response_generator(
    llama_queue, tokenizer, config, req, device, json_mode
) -> partial:
    """
    Get the correct response generator based on the request.
    """
    if not req.streaming:
        return partial(response_generator, req, llama_queue, tokenizer, config, device)
    else:
        return partial(
            streaming_generator, req, llama_queue, tokenizer, config, device, json_mode
        )
