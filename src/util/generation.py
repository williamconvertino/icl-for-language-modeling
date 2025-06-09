import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from src.util import parse_experiment_args, resolve_checkpoint_path, LightningWrapper

def nucleus_sample(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumulative_probs > top_p).float().argmax(dim=-1)
    mask = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0) > cutoff.unsqueeze(1)
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)

def generate_text(model, tokenizer, prompt, max_length, top_p, temperature):
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_tensor.clone()

    for _ in range(max_length):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = nucleus_sample(next_token_logits, top_p=top_p, temperature=temperature)
        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    # Strip prompt from output
    generated_ids = generated[0].tolist()[len(prompt_ids):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
