from transformers import AutoModelForCausalLM, AutoTokenizer

from fish_speech.text.symbols import en_symbols, jp_symbols, zh_symbols

# reuse the tokenizer from the llama
model_type = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_type)

# new tokens
new_tokens = list(set(zh_symbols + jp_symbols + en_symbols))
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({"pad_token": "<pad>"})

# pad token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

length = len(tokenizer)
if length % 8 != 0:
    length += 8 - (length % 8)

print(f"Vocab size: {len(tokenizer)}, padded to {length}")

# model = AutoModelForCausalLM.from_pretrained(
#     "fishaudio/speech-lm-300m", revision="mqtts-proto"
# )

# Resize the token embeddings to include the new tokens
# Make sure it's a multiple of 8 for faster training
# model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params / 1e6:.2f}M")

# Try tokenizing a new sequence
sequence = "[INST] Test uang1 iang5 AA an 你好 [/INST]<s>[PAD]</s>"
encoded = tokenizer.encode(sequence)
print("Test encoding....")
print(f"\tSentence: {sequence}")
print(f"\tEncoded: {encoded}")
print(f"\tDecoded: {tokenizer.batch_decode(encoded)}")

# model.push_to_hub(
#     "fishaudio/speech-lm-300m", private=True, revision="text-pretrain-10k-phones"
# )
tokenizer.push_to_hub("fishaudio/speech-lm-v1", private=True)
