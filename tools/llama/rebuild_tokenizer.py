from transformers import AutoModelForCausalLM, AutoTokenizer

from fish_speech.text.symbols import en_symbols, jp_symbols, zh_symbols

# reuse the tokenizer from the llama
model_type = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_type)

# new tokens
new_tokens = [f"<semantic_{i}>" for i in range(4096)] + list(
    set(zh_symbols + jp_symbols + en_symbols)
)
tokenizer.add_tokens(new_tokens)

# pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Vocab size: {len(tokenizer)}")

model = AutoModelForCausalLM.from_pretrained(
    "fishaudio/speech-lm-300m", revision="text-pretrain-10k"
)

# Resize the token embeddings to include the new tokens
# Make sure it's a multiple of 8 for faster training
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Try tokenizing a new sequence
sequence = "Test <semantic_0> <semantic_1023> </s> uang1 iang5 AA an"
encoded = tokenizer.encode(sequence)
print("Test encoding....")
print(f"\tSentence: {sequence}")
print(f"\tEncoded: {encoded}")
print(f"\tDecoded: {tokenizer.batch_decode(encoded)}")

model.push_to_hub(
    "fishaudio/speech-lm-300m", private=True, revision="text-pretrain-10k-phones"
)
tokenizer.push_to_hub(
    "fishaudio/speech-lm-300m", private=True, revision="text-pretrain-10k-phones"
)
