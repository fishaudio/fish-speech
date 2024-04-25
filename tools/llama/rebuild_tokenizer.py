from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Don't train the tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=0,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=[
        "<|begin_of_sequence|>",
        "<|end_of_sequence|>",
        "<|im_start|>",
        "<|im_sep|>",  # system, user, assistant, etc.
        "<|im_end|>",
        "<|semantic|>",  # audio features
        "<|pad|>",
    ],
)

# <|im_start|>user<|im_sep|>...<|im_end|>
# <|im_start|>assistant<|im_sep|><|semantic|><|semantic|><|semantic|><|semantic|><|semantic|><|im_end|>
tokenizer.train_from_iterator([], trainer=trainer)

print(len(tokenizer.get_vocab()))
x = tokenizer.encode(
    "Hello, how are you? dfgnviadfjoiviouajeiodfjv 你好世界 🈶<|semantic|>"
).ids
print(x, len(x))
print(tokenizer.decode(x, skip_special_tokens=True))


tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<|pad|>",
    bos_token="<|begin_of_sequence|>",
    eos_token="<|end_of_sequence|>",
)

# Try tokenizing a new sequence
sequence = "All around, too, lay vast quantities of the costliest merchandise, and treasures were heaped in every cranny of the rocks, but all these things only added to the desolation of the scene. 测试中文, 你好世界 🈶<|semantic|>"
encoded = tokenizer(sequence).input_ids

print("Test encoding....")
print(f"\tSentence: {sequence}")
print(f"\tEncoded: {encoded}")
print(f"\tDecoded: {tokenizer.batch_decode(encoded)}")
print(f"\tDecoded: {tokenizer.decode(encoded)}")

tokenizer.push_to_hub("fishaudio/fish-speech-1", private=True)
