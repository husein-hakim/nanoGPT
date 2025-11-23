from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders, Regex

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", 
    "<|user_end|>",
    "<|assistant_start|>", 
    "<|assistant_end|>"
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenier = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = Tokenizer(
            BPE(
                byte_fallback=True,
                unk_token=None,
                fuse_unk=False
            )
        )
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                    pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
                ])
        
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )

        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
    
    def get_vocab_size(self):
        return self.tokenier.get_vocab_size()
    
    def get_special_tokens(self):
        special_tokens_map = self.tokenier.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens
    
    def id_to_token(self, id):
        return self.tokenier.id_to_token(id)
    
    def encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids
    
    def encode_special(self, token):
        return self.tokenier.token_to_id(token)
    
    def get_bos_id(self):
        return self.encode_special("<|bos|>")
    
    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
    
