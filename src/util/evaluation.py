import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .checkpoints import resolve_checkpoint_path
from .lightning import LightningWrapper
    
@torch.no_grad()
def nucleus_sampling(logits, temperature=1.0, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0

    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, sampled)

@torch.no_grad()
def sample_tokens(model, input_ids, max_gen_len, eos_token_id, temperature=1.0, top_p=0.9):
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_gen_len):
        logits = model(generated)[:, -1, :]
        next_token = nucleus_sampling(logits, temperature, top_p)
        generated = torch.cat([generated, next_token], dim=1)

        if (next_token == eos_token_id).all():
            break

    return generated

def eval_generation(model, args, splits, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt_path = resolve_checkpoint_path(model.config.get_name(), args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    
    lightning_model = LightningWrapper.load_from_checkpoint(
        ckpt_path,
        model=model,
        tokenizer=tokenizer,
        args=args,
    ).to(device).eval()

    output_dir = os.path.join(args.output_dir, "generations", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model.config.get_name()}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(args.num_samples), desc="Generating"):
            input_ids = splits["test"][i][:model.config.max_seq_len]
            input_ids = input_ids.unsqueeze(0).to(device)

            prompt_len = min(input_ids.shape[1] // 2, model.config.max_seq_len // 2)
            prompt = input_ids[:, :prompt_len]

            generated = sample_tokens(
                lightning_model,
                prompt,
                args.max_length,
                eos_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                top_p=args.p_value
            )[0].tolist()

            decoded_full = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            decoded_prompt = tokenizer.decode(prompt[0].tolist(), skip_special_tokens=True)
            decoded_gen = tokenizer.decode(generated[len(prompt[0]):], skip_special_tokens=True)

            f.write("============\n")
            f.write("ORIGINAL SAMPLE:\n")
            f.write("============\n")
            f.write(decoded_full + "\n")
            f.write("============\n")
            f.write("GENERATION:\n")
            f.write("============\n")
            f.write(f"{decoded_prompt} [{decoded_gen}]\n")
            f.write("============\n")
            f.write("++++++++++++\n")