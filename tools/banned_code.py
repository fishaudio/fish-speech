from fish_speech.conversation import Conversation, Message

def execute_request(
    input_queue: queue.Queue,
    tokenizer: FishTokenizer,
    config: BaseModelArgs,
    request: ServeRequest,
    device: str = "cuda:0",
):

    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)
    messages = []
    for message in request.messages:
        messages.append(message.to_conversation_message())

    assert len(messages) >= 1, "At least one message is required"
    # assert messages[-1].role == "user", "The last message must be from the user"

    if messages[-1].role == "user":
        messages.append(
            Message(role="assistant", parts=[], add_im_end=False, modality="voice")
        )
    elif messages[-1].role == "raw":
        messages[-1].add_im_start = False
        messages[-1].add_im_end = False
        messages[-1].modality = "voice"
    else:
        assert (
            messages[-1].role == "assistant"
        ), "The last message must be from the assistant"
        messages[-1].add_im_end = False

    conv = Conversation(messages=messages)

    # conv.visualize(tokenizer)
    prompt = conv.encode_for_inference(
        tokenizer=tokenizer, num_codebooks=config.num_codebooks
    ).to(device)

    if request.streaming:
        for i in range(request.num_samples):
            yield ServeStreamResponse(
                sample_id=i,
                delta=ServeStreamDelta(
                    role="assistant",
                ),
            )

    req = {
        "prompt": prompt,
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }

    start = time.time()
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))

    # Decoding
    decode_buffer = [[] for _ in range(request.num_samples)]
    parts = [[] for _ in range(request.num_samples)]

    def send_reset_buffer(sample_id):
        nonlocal decode_buffer
        if len(decode_buffer[sample_id]) == 0:
            return

        decoded = tokenizer.decode(decode_buffer[sample_id])
        part = ServeTextPart(text=decoded)

        if request.streaming:
            yield ServeStreamResponse(delta=ServeStreamDelta(part=part))
        else:
            parts[sample_id].append(part)

        decode_buffer[sample_id] = []

    # Decode process
    finished = [False for _ in range(request.num_samples)]
    stats = {}
    idx = 0
    while True:
        response = response_queue.get()

        if response in ["stop", "error"]:
            break

        for sample_id, tokens in enumerate(response):
            if finished[sample_id]:
                continue

            if tokens[0] == im_end_id:
                finished[sample_id] = True
                if request.streaming:
                    yield from send_reset_buffer(sample_id)
                    yield ServeStreamResponse(
                        sample_id=sample_id,
                        finish_reason="stop",
                        stats=stats,
                    )
                continue

            is_semantic = (
                tokenizer.semantic_begin_id <= tokens[0] <= tokenizer.semantic_end_id
            )
            if is_semantic and request.streaming:
                yield from send_reset_buffer(sample_id)
                # Streaming vq
                _tokens = tokens[1:].clone()

                if config.share_codebook_embeddings is False:
                    for i in range(len(_tokens)):
                        _tokens[i] -= config.codebook_size * i

                yield ServeStreamResponse(
                    sample_id=sample_id,
                    delta=ServeStreamDelta(part=ServeVQPart(codes=_tokens.tolist())),
                )
                continue

            # Not streaming vq
            if is_semantic:
                yield from send_reset_buffer(sample_id)
                # None streaming vq
                if len(parts[sample_id]) == 0 or not isinstance(
                    parts[sample_id][-1], ServeVQPart
                ):
                    _tokens = tokens[1:].clone()

                    if config.share_codebook_embeddings is False:
                        for i in range(len(_tokens)):
                            _tokens[i] -= config.codebook_size * i

                    parts[sample_id].append(ServeVQPart(codes=_tokens.tolist()))
                else:
                    for codebook_id, value in enumerate(tokens[1:, :]):
                        val = value.item()
                        if config.share_codebook_embeddings is False:
                            val -= config.codebook_size * codebook_id

                        parts[sample_id][-1].codes[codebook_id].append(val)
                continue

            if not is_semantic:
                # Stream text decode is not supported now
                decode_buffer[sample_id].append(tokens[0, 0])

        if idx == 0:
            stats["time_to_first_token"] = (time.time() - start) * 1000

        idx += 1

    for sample_id in range(request.num_samples):
        yield from send_reset_buffer(sample_id)

    stats["total_time"] = (time.time() - start) * 1000
    stats["total_tokens"] = idx

    if request.streaming:
        for sample_id in range(request.num_samples):
            if finished[sample_id]:
                continue
            yield ServeStreamResponse(
                finish_reason=response, stats=stats, sample_id=sample_id
            )
        return

    yield ServeResponse(
        messages=[
            ServeMessage(role="assistant", parts=parts[i])
            for i in range(request.num_samples)
        ],
        finish_reason=response,
        stats=stats,
    )


@routes.http.post("/v1/chat")
def api_invoke_chat(
    req: Annotated[ServeRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """

    # This makes torch compile happy
    assert (
        req.num_samples == GLOBAL_NUM_SAMPLES
    ), f"num_samples must be {GLOBAL_NUM_SAMPLES}"

    content_type = request.headers.get("Content-Type", "application/json")
    json_mode = "application/json" in content_type

    async def wrapped_generator():
        generator = execute_request(llama_queue, tokenizer, config, req, args.device)

        for i in generator:
            if json_mode:
                body = i.model_dump_json().encode("utf-8")
                yield b"data: " + body + b"\n\n"
            else:
                body = ormsgpack.packb(i, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
                yield struct.pack("I", len(body)) + body

    # Naive mode
    if req.streaming is False:
        result = next(execute_request(llama_queue, tokenizer, config, req, args.device))

        if json_mode:
            return JSONResponse(result.model_dump())
        else:
            return ormsgpack.packb(result, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)

    return StreamResponse(
        iterable=wrapped_generator(), content_type="text/event-stream"
    )