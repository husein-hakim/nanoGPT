import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
from tokenizer import CustomTokenizer
from dataloader import get_dataloader

def test_dataloader():
    print("Loading tokenizer...")
    tokenizer = CustomTokenizer

    print("Initializing DataLoader...")
    loader = get_dataloader(tokenizer, batch_size=4, context_length=128)

    print("Fetching first batch (this might take a moment to stream)...")
    start_time = time.time()
    
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("❌ Error: Loader is empty!")
        return

    end_time = time.time()
    print(f"✅ First batch fetched in {end_time - start_time:.2f} seconds.")

    input_ids = batch["input_ids"]
    
    print(f"\n--- Tensor Stats ---")
    print(f"Shape: {input_ids.shape}") 
    # EXPECTED: torch.Size([4, 128]) -> [batch_size, context_length]
    
    print(f"Dtype: {input_ids.dtype}") 
    # EXPECTED: torch.int64 (or int32) - NOT float
    
    print(f"Device: {input_ids.device}") 
    # EXPECTED: cpu (Data loading happens on CPU, moved to GPU in train loop)

    # 5. VISUAL INSPECTION (Decode back to text)
    print(f"\n--- Decoding First Sample ---")
    # We take the first row [0] and turn it back into words
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    print("RAW TEXT START:")
    print("-" * 20)
    print(decoded_text[:500]) # Print first 500 chars
    print("-" * 20)

    # 6. Check for Special Tokens
    print(f"\n--- Token Checks ---")
    print(f"Starts with BOS/CLS? {input_ids[0][0]}") 
    # Some models require a specific start token ID
    
    print(f"Contains Padding? {tokenizer.pad_token_id in input_ids[0]}")
    # If using 'max_length' padding, you might see padding at the end. 
    # If using 'packing' (concatenating samples), you shouldn't see much padding.

if __name__ == "__main__":
    test_dataloader()