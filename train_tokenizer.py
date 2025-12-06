import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer

# Import your class
from tokenizer import CustomTokenizer 

# --- Configuration ---
VOCAB_SIZE = 65536 
DOC_CAP = 10_000  
MAX_CHARS = 10_000_000
SAVE_DIR = "./nanogpt_tokenizer"

def get_training_corpus():
    """
    Streams from HuggingFace, caps document length, and yields text.
    """
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    
    nchars = 0
    print(f"Streaming data (Stop limit: {MAX_CHARS:,} chars)...")
    
    for item in dataset:
        text = item["text"]
        
        # Cap huge documents
        if len(text) > DOC_CAP:
            text = text[:DOC_CAP]
            
        nchars += len(text)
        yield text
        
        if nchars >= MAX_CHARS:
            print(f"Reached {nchars} characters. Stopping.")
            break

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    t0 = time.time()
    tokenizer = CustomTokenizer.train_from_iterator(get_training_corpus(), vocab_size=VOCAB_SIZE)
    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f}s")

    tokenizer.save(SAVE_DIR)
    
    print("Calculating token bytes for BPB metric...")
    
    vocab_size = tokenizer.get_vocab_size()
    
    special_set = set(tokenizer.get_special_tokens())
    
    token_bytes = []
    
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        
        # If it's a special token (like <|pad|> or <|bos|>), it counts as 0 bytes
        if token_str in special_set or token_str == "":
            token_bytes.append(0)
        else:
            # Count UTF-8 bytes (e.g., 'Apple' = 5, '😊' = 4)
            token_bytes.append(len(token_str.encode("utf-8")))
            
    # Convert to Tensor and Save
    token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32)
    torch.save(token_bytes_tensor, os.path.join(SAVE_DIR, "token_bytes.pt"))
    
    print(f"Saved token_bytes.pt to {SAVE_DIR}")
    
    print("\n--- Sanity Check ---")
    print(f"Vocab Size: {vocab_size}")
    print(f"Token Bytes Shape: {token_bytes_tensor.shape}")
    sample_text = "Hello world"
    ids = tokenizer.encode(sample_text)
    print(f"'{sample_text}' -> {ids}")
    print(f"Back to text -> '{tokenizer.decode(ids)}'")

if __name__ == "__main__":
    main()