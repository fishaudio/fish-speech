from transformers import LlamaModel, LlamaConfig, AutoTokenizer

# reuse the tokenizer from the llama
model_type = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_type)

# new tokens
new_tokens = [f"<semantic_{i}>" for i in range(4096)]
tokenizer.add_tokens(new_tokens + ["<pad>"])

# pad token
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

print(f"Vocab size: {len(tokenizer)}")

hidden_size = 1024
intermediate_size = hidden_size * (11 / 3)
# then round to the nearest multiple of 8
intermediate_size = round(intermediate_size / 8) * 8
print(f"Hidden size: {hidden_size}")
print(f"Intermediate size: {intermediate_size}")

model = LlamaModel(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=20,
        num_attention_heads=16,
        max_position_embeddings=4096,
    )
)

model = model.bfloat16()

# Resize the token embeddings to include the new tokens
# Make sure it's a multiple of 8 for faster training
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Try tokenizing a new sequence
sequence = "Test <semantic_0> <semantic_1023> <pad>"
encoded = tokenizer.encode(sequence)
print("Test encoding....")
print(f"\tSentence: {sequence}")
print(f"\tEncoded: {encoded}")
print(f"\tDecoded: {tokenizer.batch_decode(encoded)}")

# model.save_pretrained("./checkpoints/speech-lm-300m-init")
# tokenizer.save_pretrained("./checkpoints/speech-lm-300m-init")

model.push_to_hub("fishaudio/speech-lm-300m", private=True, revision="init")
tokenizer.push_to_hub("fishaudio/speech-lm-300m", private=True, revision="init")
