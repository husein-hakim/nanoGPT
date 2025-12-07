import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizer import CustomTokenizer
from dataloader import get_dataloader, get_sanity_loader
from GPTModel import Config, GPT
from transformers import get_cosine_schedule_with_warmup
import time

out_dir = 'pretrain'
os.makedirs(out_dir, exist_ok=True)

# total_batch_size = 524288
total_batch_size = 4096
device_batch_size = 4
block_size = 1024
# max_iters = 5000
max_iters = 500
learning_rate = 1e-3
warmup_steps = 100     
# eval_interval = 200     
eval_interval = 50  
save_interval = 500

if torch.backends.mps.is_available():
    device = "mps"
    print("using mps")
elif torch.cuda.is_available():
    device = "cuda"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

#Gradient Accumulation Setup
grad_accum_steps = total_batch_size // (device_batch_size*block_size)

#Tokenizer Setup
print(f'Loading tokenizer')
backend = Tokenizer.from_file('nanogpt_tokenizer/tokenizer.json')
tokenizer = CustomTokenizer(backend)

#Dataloader setup
print('Loading Dataloader')
train_loader = get_dataloader(tokenizer, device_batch_size, block_size)
# train_loader = get_sanity_loader(tokenizer, device_batch_size, block_size)
train_iter = iter(train_loader)

#Initialize model
print('Initializing Model')
model = GPT(Config)
model.to(device)

#Setting optimizer and lr
optimizer = model.configure_optimizers(1e-1, learning_rate, betas=(0.9, 0.95))
# scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_iters)

def get_batch():
    global train_loader, train_iter
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    input_ids = batch['input_ids']
    x = input_ids[:, :-1].to(device)
    y = input_ids[:, 1:].to(device)

    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    losses = torch.zeros(10)
    for i in range(10):
        X, y = get_batch()
        logits, loss = model(X, y)
        losses[i] = loss.item()

    model.train()
    return losses.mean()

#Training Loop
print(f"Starting training for {max_iters} steps...")
t0 = time.time()

for epoch in range(max_iters):
    optimizer.zero_grad()
    loss_accum = 0.0

    for step in range(grad_accum_steps):
        X, y = get_batch()
        logits, loss = model(X, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.item()
        loss.backward()

    optimizer.step()
    # scheduler.step()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss_accum:.4f} | time: {dt*1000:.2f}ms')

    if epoch % eval_interval == 0:
        val_loss = estimate_loss()
        model.eval()
        context = "The king said"
        ids = tokenizer.encode_one(context, prepend="<|bos|>")
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        y = model.generate(x, max_new_tokens=50, temperature=1.0, top_k=25)
        print(f"Val: step {epoch} | val_loss {val_loss:.4f}")
        print(f'Ouput: {tokenizer.decode(y[0].tolist())}')
        model.train()

    if epoch > 0 and epoch % save_interval == 0:
        checkpoint_path = os.path.join(out_dir, f'ckpt_{step}.pt')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': Config,
            'step': epoch,
            'val_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")

print('Training Complete')
