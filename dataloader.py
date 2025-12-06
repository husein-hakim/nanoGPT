import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_dataloader(tokenizer, batch_size=8, context_length=1024):
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=72)

    def tokenizer_preprocess(examples):
        print(f"Type of text input: {type(examples['text'])}")
        all_input_ids = []
        for text in examples['text']:
            ids = tokenizer.encode_one(text, prepend='<|bos|>')
            if len(ids) >= context_length:
                ids = ids[:context_length]
            elif len(ids) < context_length:
                ids = ids + [tokenizer.encode_special("<|pad|>")]*(context_length-len(ids))
            assert len(ids) == context_length, f"Error: Got length {len(ids)}"
            all_input_ids.append(ids)
        return {"input_ids": all_input_ids}
    
    dataset = dataset.map(tokenizer_preprocess, remove_columns=["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"], batched=True)
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def get_sanity_loader(tokenizer, batch_size=4, context_length=1024):
    with open('input.txt', 'r') as f:
        text = f.read()

    tokens = tokenizer.encode_one(text, prepend='<|bos|>')
    dataset = torch.tensor(tokens, dtype=torch.float16)
    print(f'Total Tokens: {data.shape}')
    
    class SanityIterator:
        def __init__(self, data, batch_size, context_length):
            self.data = data
            self.batch_size = batch_size
            self.block_size = context_length
            self.n = len(data)

        def __iter__(self):
            return self

        def __next__(self):
            ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
            x = torch.stack([self.data[i : i + self.block_size] for i in ix])
            return {"input_ids": x}

    return SanityIterator(data, batch_size, context_length)