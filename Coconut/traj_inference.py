import argparse
import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import CoconutGPT_Same_Word_Embedding
from data import _load_trajectory_dataset


@torch.no_grad()
def greedy_decode_from_embeds(
    lm,
    tokenizer,
    prefix_embeds,          # (1, Lp, H)  — optional prompt suffix appended AFTER latent
    latent_embeds,          # (1, Ll, H)
    max_new_tokens=128,
):
    """Decode text from latent thought embeddings.

    The latent embeddings come first (matching the training layout in coconut.py:786
    where `[continuous_embeds, other_embeds]` = `[latent, text]`).
    prefix_embeds are appended after latent as an optional steering suffix.
    position_ids start at 1 to match training (coconut.py:791 uses arange(1, len+1)).
    """
    device = next(lm.parameters()).device
    eos = tokenizer.eos_token_id

    # latent first, then optional prefix — mirrors training order [latent, text]
    cur = torch.cat([latent_embeds, prefix_embeds], dim=1).to(device)
    generated = []

    for _ in range(max_new_tokens):
        attn = torch.ones(cur.shape[:2], device=device, dtype=torch.long)
        # position_ids start at 1 to match training (coconut.py uses arange(1, len+1))
        pos = torch.arange(1, cur.shape[1] + 1, device=device, dtype=torch.long).unsqueeze(0)

        out = lm(inputs_embeds=cur, attention_mask=attn, position_ids=pos)
        next_id = int(torch.argmax(out.logits[0, -1], dim=-1).item())
        if next_id == eos:
            break

        generated.append(next_id)
        next_embed = lm.get_input_embeddings()(torch.tensor([[next_id]], device=device))
        cur = torch.cat([cur, next_embed], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def pick_sample(ds, sample_id=None, sample_index=None):
    if sample_id is not None:
        for i in range(len(ds)):
            if ds[i].get("instance_id") == sample_id:
                return ds[i]
        raise ValueError(f"sample_id '{sample_id}' not found. Total samples: {len(ds)}")

    if sample_index is None:
        sample_index = 0
    return ds[int(sample_index)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)

    # Trajectory dataset directory with *.traj.json
    ap.add_argument("--traj_dir", type=str, required=True)
    ap.add_argument("--sample_id", type=str, default=None, help="e.g. 'instance123-step0'")
    ap.add_argument("--sample_index", type=int, default=None)

    # Latent params
    ap.add_argument("--scheduled_stage", type=int, default=2)
    ap.add_argument("--max_latent_stage", type=int, default=5)
    ap.add_argument("--c_thought", type=int, default=2)

    # Generation params
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # Latent decoding params
    ap.add_argument("--decode_latents", action="store_true")
    ap.add_argument("--decode_tokens", type=int, default=128)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # --- tokenizer + special tokens (must match training) ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # This token is used in training code for init + passed into wrapper (step_start_id)
    step_start_id = tokenizer.convert_tokens_to_ids("<<")
    if step_start_id is None or step_start_id < 0:
        step_start_id = tokenizer.eos_token_id

    # --- base + explainable models ---
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    explainable_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    # Resize embeddings because we added new tokens.
    # Only base_model is resized — matches how the checkpoint was saved during training.
    # explainable_model keeps its original vocab size (e.g. 151936) because:
    #   - Training code never resized it, so checkpoint expainable_llm is at original size.
    #   - All new special token IDs (latent, start-latent, end-latent) are assigned into
    #     the range of the original vocab (< original vocab_size), so they're in range.
    base_model.resize_token_embeddings(len(tokenizer))

    # Init new tokens from a known token (same idea as training)
    emb = base_model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    if target_id is None or target_id < 0:
        target_id = tokenizer.eos_token_id

    with torch.no_grad():
        for tid in [latent_id, start_id, end_id]:
            emb.weight.data[tid] = emb.weight.data[target_id]
            if hasattr(base_model, "lm_head"):
                base_model.lm_head.weight.data[tid] = base_model.lm_head.weight.data[target_id]

    # Minimal config object the wrapper expects
    cfg = SimpleNamespace(
        training_method="full",
        cthought=args.c_thought,
        maxlatentstage=args.max_latent_stage,
        visualize=False,
        wprompt=False,
        explainmode="v1_aug",
        packing=False,
    )

    model = CoconutGPT_Same_Word_Embedding(
        base_model,
        explainable_model,
        tokenizer,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
        step_start_id,
        args.c_thought,
        cfg,
    ).to(device)

    # --- load checkpoint ---
    sd = torch.load(args.ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    explainable_model.eval()

    # --- pick a sample from trajectory dataset ---
    ds = _load_trajectory_dataset(args.traj_dir)
    sample = pick_sample(ds, sample_id=args.sample_id, sample_index=args.sample_index)

    prompt = sample["prompt"]
    target = sample.get("patch", "")

    print("\n" + "=" * 80)
    print("SAMPLE instance_id:", sample.get("instance_id"))
    print("=" * 80)
    print("PROMPT (first 2000 chars):\n", prompt[-1000:])
    print("\nTARGET command/patch:\n", target[:2000])
    print("=" * 80)

    # --- build input_ids with latent tokens ---
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    k = min(args.scheduled_stage, args.max_latent_stage) * args.c_thought

    input_ids = torch.tensor(
        [prompt_ids + [start_id] + [latent_id] * k + [end_id]],
        dtype=torch.long,
        device=device,
    )
    attn = torch.ones_like(input_ids, device=device)
    pos = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)

    # --- forward once to extract latent embeddings ---
    out = model.forward(input_ids, attn, labels=input_ids.clone(), position_ids=pos)
    inputs_embeds = out.inputs_embeds  # (1, seq, hidden)

    latent_pos = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0].tolist()
    print("\nLATENTS:")
    print("latent_id:", latent_id, "k:", k, "found:", len(latent_pos))
    print("latent positions head:", latent_pos[:20])

    # --- generate model output (command) ---
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=args.max_new_tokens,
        synced_gpus=False,
    )
    gen_ids = gen_ids[0].tolist()
    gen_text = tokenizer.decode(gen_ids[input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "-" * 80)
    print("MODEL OUTPUT (decoded):")
    print(gen_text)
    print("-" * 80)

    # --- decode latent thoughts via explainable model ---
    if args.decode_latents:
        if len(latent_pos) == 0:
            print("No latent tokens found => nothing to decode.")
            return

        # embeddings at latent positions: (1, k, H)
        lat_emb = inputs_embeds[:, latent_pos, :]

        n_thoughts = max(1, k // args.c_thought) if k > 0 else 0
        print("\nDECODE LATENTS:")
        print("n_thoughts:", n_thoughts, "c_thought:", args.c_thought)

        for ti in range(n_thoughts):
            s = ti * args.c_thought
            e = (ti + 1) * args.c_thought
            chunk = lat_emb[:, s:e, :]

            prefix = f"Step {ti + 1}: "
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            prefix_emb = explainable_model.get_input_embeddings()(
                torch.tensor([prefix_ids], device=device)
            )

            decoded = greedy_decode_from_embeds(
                explainable_model,
                tokenizer,
                prefix_embeds=prefix_emb,
                latent_embeds=chunk,
                max_new_tokens=args.decode_tokens,
            )

            print("\n" + "-" * 80)
            print(f"LATENT THOUGHT #{ti + 1} (decoded):")
            print(decoded)
            print("-" * 80)


if __name__ == "__main__":
    main()
