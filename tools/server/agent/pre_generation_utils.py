import queue

from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.inference import GenerateRequest
from fish_speech.tokenizer import IM_END_TOKEN


def prepare_messages(request, tokenizer, config):
    """
    Reorganise the provided list of messages into a conversation.
    Encode the conversation for inference.
    """
    # Convert the messages to ConversationMessage objects
    messages = [msg.to_conversation_message() for msg in request.messages]

    if len(messages) < 1:
        raise ValueError("At least one message is required")

    # Check the last message to determine the next step
    last_role = messages[-1].role
    match last_role:
        case "user":
            # The last message is from the user, ask the assistant to respond with a new message
            messages.append(
                Message(role="assistant", parts=[], add_im_end=False, modality="voice")
            )
        case "raw":
            # The last message is raw text, ask the assistant to complete it
            messages[-1].add_im_start = False
            messages[-1].add_im_end = False
            messages[-1].modality = "voice"
        case "assistant":
            # The last message is from the assistant, ask the assistant to continue
            messages[-1].add_im_end = False
        case _:
            # We expect it to be assistant if not user or raw
            raise ValueError("The last message must be from the assistant, user or raw")

    # Create a conversation object and encode it for inference
    conv = Conversation(messages=messages)
    prompt = conv.encode_for_inference(
        tokenizer=tokenizer, num_codebooks=config.num_codebooks
    )
    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)

    return prompt, im_end_id


def create_generation_request(prompt, request, im_end_id, device):
    """
    Convert the request into a dictionary that can be sent to the model for generation.
    """
    req = {
        "prompt": prompt.to(device),
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }
    return req


def send_generation_request(input_queue, req):
    """
    Send the generation request to the model and return a queue to get the response.
    """
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))
    return response_queue
