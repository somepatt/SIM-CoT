# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import functools
import gc
import json
import math
import os
import sys
from copy import copy

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import wandb

from coconut import Coconut, CoconutGPT_Same_Word_Embedding
import data as traj_data  # <-- your dataset pipeline
from utils import Config, set_seed


def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print(f"{name} requires gradient")


def save_jsonl_line(filepath, data):
    if not isinstance(data, dict):
        raise ValueError("data 必须是一个字典")
    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")


def _move_batch_to_device(batch, device):
    """
    Move only tensors to GPU. Keep python objects (e.g., list-of-lists steps).
    """
    moved = {}
    for k, v in batch.items():
        if k == "idx":
            moved[k] = v
        elif torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _report_nonfinite_params(module, module_name, max_items=8):
    total_bad = 0
    bad_items = []
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is None:
                continue
            bad_mask = ~torch.isfinite(param.data)
            bad_count = int(bad_mask.sum().item())
            if bad_count > 0:
                total_bad += bad_count
                if len(bad_items) < max_items:
                    bad_items.append((name, bad_count))
    if total_bad > 0:
        print(f"[nonfinite_params] {module_name}: total_bad={total_bad}, samples={bad_items}")
    else:
        print(f"[nonfinite_params] {module_name}: OK")
    return total_bad


def _module_grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.numel() == 0:
            continue
        norm = grad.norm(2)
        if not torch.isfinite(norm):
            return float("nan")
        total += float(norm.item()) ** 2
    return total ** 0.5


def _grad_health_report(module, max_items=8):
    total_bad = 0
    bad_items = []
    max_abs_grad = 0.0

    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        bad = int((~finite).sum().item())
        if bad > 0:
            total_bad += bad
            if len(bad_items) < max_items:
                bad_items.append((name, bad))
            continue

        if grad.numel() > 0:
            cur_max = float(grad.abs().max().item())
            if cur_max > max_abs_grad:
                max_abs_grad = cur_max

    return {
        "total_bad": total_bad,
        "bad_items": bad_items,
        "max_abs_grad": max_abs_grad,
    }


def _sanitize_token_rows(base_lm, token_ids, fallback_id):
    emb = base_lm.get_input_embeddings().weight.data
    lm_head = base_lm.lm_head.weight.data if hasattr(base_lm, "lm_head") else None

    def safe_row(table, idx):
        if idx is None or idx < 0 or idx >= table.size(0):
            return torch.zeros_like(table[0])
        row = table[idx]
        if torch.isfinite(row).all():
            return row.clone()
        return torch.zeros_like(row)

    fallback_emb = safe_row(emb, fallback_id)
    fallback_lm = safe_row(lm_head, fallback_id) if lm_head is not None else None

    repaired = 0
    for tid in token_ids:
        if tid is None or tid < 0 or tid >= emb.size(0):
            continue
        if not torch.isfinite(emb[tid]).all():
            emb[tid].copy_(fallback_emb)
            repaired += 1
        if lm_head is not None and not torch.isfinite(lm_head[tid]).all():
            lm_head[tid].copy_(fallback_lm)
            repaired += 1
    return repaired


def _sanitize_special_embeddings(model, token_ids, fallback_id):
    repaired_total = 0
    with torch.no_grad():
        if hasattr(model, "base_causallm"):
            repaired_total += _sanitize_token_rows(model.base_causallm, token_ids, fallback_id)
        elif hasattr(model, "get_input_embeddings"):
            repaired_total += _sanitize_token_rows(model, token_ids, fallback_id)

        if hasattr(model, "expainable_llm"):
            repaired_total += _sanitize_token_rows(model.expainable_llm, token_ids, fallback_id)
    print(f"[sanitize_special_embeddings] repaired_rows={repaired_total}")
    return repaired_total


def _build_explainable_ids_list(raw_steps, input_ids, latent_id, c_thought, l_id, r_id):
    """
    Build explainable supervision from the most recent thought steps only.
    The number of selected thought steps matches latent thought slots per sample.
    """
    explainable = []
    c_thought = max(int(c_thought), 1)

    for sample_steps, sample_input_ids in zip(raw_steps, input_ids):
        latent_count = int((sample_input_ids == latent_id).sum().item())
        n_thoughts = latent_count // c_thought

        if n_thoughts > 0:
            selected_steps = sample_steps[-n_thoughts:]
        else:
            selected_steps = []

        flat = []
        for step in selected_steps:
            flat.extend([l_id] + list(step) + [r_id])
        explainable.append(flat)

    return explainable


def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    raw_cfg = config_dict  # pass dict into data.py helpers (expects .get())
    auto_resume = bool(raw_cfg.get("auto_resume", True))
    attn_implementation = raw_cfg.get("attn_implementation", None)
    force_math_sdp = bool(raw_cfg.get("force_math_sdp", False))

    if force_math_sdp and torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        if rank == 0:
            print("Forced CUDA SDPA backend to math (flash/mem_efficient disabled).")

    set_seed(configs.seed)

    save_dir = os.path.join(configs.save_path, configs.name)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    dist.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.
    if auto_resume and len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print(
                "Warning: found previous run and gonna resume from that. "
                "the inputted `resume` argument is ignored!"
            )
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        # configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        configs.load_model_path = load_dir
        if rank == 0:
            print(f"Loading from previous run epoch_{configs.resume}!")
    elif configs.resume != 0:
        if configs.load_model_path == "None":
            if rank == 0:
                print(
                    f"Warning: you want to skip the first {configs.resume} but "
                    f"you are not loading any existing checkpoint!"
                )
        if rank == 0:
            print(
                f"Loading from {configs.load_model_path} and skip the first "
                f"{configs.resume} epochs"
            )
    elif (not auto_resume) and (len(cur_ckpts) > 0) and rank == 0:
        print("Auto-resume is disabled (auto_resume=false), ignoring existing checkpoints in save_dir.")

    trust_remote = bool(raw_cfg.get("trust_remote_code", False))

    model_load_kwargs = {"trust_remote_code": trust_remote}
    if attn_implementation:
        model_load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            **model_load_kwargs,
        ).to(local_rank)
    except TypeError:
        # Some transformers/model versions may not support attn_implementation.
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            trust_remote_code=trust_remote,
        ).to(local_rank)

    if configs.mode != "coconut_baseline":
        try:
            explainable_model = AutoModelForCausalLM.from_pretrained(
                configs.model_id,
                **model_load_kwargs,
            ).to(local_rank)
        except TypeError:
            explainable_model = AutoModelForCausalLM.from_pretrained(
                configs.model_id,
                trust_remote_code=trust_remote,
            ).to(local_rank)
    else:
        explainable_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id,
        trust_remote_code=trust_remote,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Coconut special tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    tokenizer.add_tokens("<<")
    tokenizer.add_tokens(">>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    l_id  = tokenizer.convert_tokens_to_ids("<<")
    r_id  = tokenizer.convert_tokens_to_ids(">>")

    loaded = False
    if configs.load_model_path != "None":
        saved_weights = torch.load(configs.load_model_path, map_location="cpu")

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif (not configs.coconut) and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run; will handle later
            pass
        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    # If we are training latent mode, we need new token embeddings.
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()

        # safer init token for Qwen-like tokenizers
        init_from = raw_cfg.get("embed_init_token", None)
        if init_from is None:
            target_id = tokenizer.eos_token_id
        else:
            target_id = tokenizer.convert_tokens_to_ids(init_from)
            if target_id is None or target_id < 0:
                target_id = tokenizer.eos_token_id

        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding

            # tied weights: update lm_head too
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        if configs.mode == "coconutgpt_same_word_embedding":
            # step_start_id is used by CoconutGPT as a prompt token for step decoding;
            # default to EOS for safety (configurable via step_start_token).
            step_start_tok = raw_cfg.get("step_start_token", None)
            if step_start_tok is None:
                step_start_id = tokenizer.eos_token_id
            else:
                step_start_id = tokenizer.convert_tokens_to_ids(step_start_tok)
                if step_start_id is None or step_start_id < 0:
                    step_start_id = tokenizer.eos_token_id

            model = CoconutGPT_Same_Word_Embedding(
                model,
                explainable_model,
                tokenizer,
                latent_id,
                start_id,
                end_id,
                tokenizer.eos_token_id,
                step_start_id,
                configs.c_thought,
                configs,
            )
        elif configs.mode == "coconut_baseline":
            model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        else:
            raise ValueError(f"don't support model {configs.mode=}")

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    # Repair possible NaN/Inf rows for special tokens after checkpoint loading.
    _sanitize_special_embeddings(
        model=model,
        token_ids=[latent_id, start_id, end_id, l_id, r_id],
        fallback_id=tokenizer.eos_token_id,
    )

    nonfinite_bad_total = 0
    if configs.mode == "coconutgpt_same_word_embedding":
        bad_base = _report_nonfinite_params(model.base_causallm, "base_causallm_after_load")
        bad_explain = _report_nonfinite_params(model.expainable_llm, "expainable_llm_after_load")
        nonfinite_bad_total = bad_base + bad_explain
    else:
        nonfinite_bad_total = _report_nonfinite_params(model, "model_after_load")

    allow_nonfinite_checkpoint = bool(raw_cfg.get("allow_nonfinite_checkpoint", False))
    if (configs.load_model_path != "None") and (nonfinite_bad_total > 0) and (not allow_nonfinite_checkpoint):
        raise RuntimeError(
            f"Loaded checkpoint contains non-finite weights (count={nonfinite_bad_total}). "
            f"Checkpoint path: {configs.load_model_path}. "
            "Use a different checkpoint or set allow_nonfinite_checkpoint: true to bypass (not recommended)."
        )

    if rank == 0:
        print(f"Running FSDP on rank = {rank}, world size = {world_size}")

    model = model.to(local_rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[local_rank])
    else:
        parallel_model = FSDP(
            model,
            auto_wrap_policy=llama_auto_wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
        )

    del model

    if rank == 0:
        print(parallel_model)
        check_requires_grad(parallel_model.module)

    # -------------------------
    # DATA (trajectory via data.py)
    # -------------------------
    skip_generation_eval = bool(raw_cfg.get("skip_generation_eval", True))
    max_new_tokens = int(raw_cfg.get("max_new_tokens", 256))
    loss_log_interval = int(raw_cfg.get("loss_log_interval", 10))
    max_grad_norm = float(raw_cfg.get("max_grad_norm", 1.0))
    grad_health_interval = int(raw_cfg.get("grad_health_interval", 1))
    detect_anomaly = bool(raw_cfg.get("detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if rank == 0:
            print("Autograd anomaly detection is enabled.")

    base_dataset_valid = traj_data.get_dataset(
        dataset_name=configs.val_path,
        split=raw_cfg.get("val_split", "train"),
        tokenizer=tokenizer,
        max_size=raw_cfg.get("max_val_size", 2000 if not configs.debug else 128),
        max_seq_length=raw_cfg.get("max_seq_length", 8192),
        steps_field=raw_cfg.get("steps_field", None),
        steps_pattern=raw_cfg.get("steps_pattern", None),
        steps_delimiter=raw_cfg.get("steps_delimiter", None),
        max_step_tokens=raw_cfg.get("max_step_tokens", 256),
        require_steps=raw_cfg.get("require_steps", False),
    )

    if not configs.only_eval:
        base_dataset_train = traj_data.get_dataset(
            dataset_name=configs.train_path,
            split=raw_cfg.get("train_split", "train"),
            tokenizer=tokenizer,
            max_size=raw_cfg.get("max_train_size", 20000 if not configs.debug else 512),
            max_seq_length=raw_cfg.get("max_seq_length", 8192),
            steps_field=raw_cfg.get("steps_field", None),
            steps_pattern=raw_cfg.get("steps_pattern", None),
            steps_delimiter=raw_cfg.get("steps_delimiter", None),
            max_step_tokens=raw_cfg.get("max_step_tokens", 256),
            require_steps=raw_cfg.get("require_steps", False),
        )

    total_train_steps = 0
    if not configs.debug and (not configs.only_eval) and configs.wandb and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None
        text_table = None

    if configs.reset_optimizer:
        optimizer = None
    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay
        )

    best_acc = 0

    collator = traj_data.MyCollator(
        tokenizer, latent_id=latent_id, label_pad_token_id=-100
    )

    for epoch in range(configs.resume, configs.num_epochs + configs.resume):
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )

        # (Optional) generation dataset for eval
        dataset_gen_val = traj_data.get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            raw_cfg,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:
            dataset_train = traj_data.get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                raw_cfg,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

        dataset_loss_val = traj_data.get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            raw_cfg,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            shuffle=False,
        )
        valid_loss_dataloader = torch.utils.data.DataLoader(
            dataset_loss_val,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_loss_val, shuffle=False),
        )

        if configs.reset_optimizer:
            if optimizer is not None:
                del optimizer
            optimizer = optim.AdamW(
                parallel_model.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay,
            )

        # -------------------------
        # TRAIN
        # -------------------------
        if not configs.only_eval:
            parallel_model.module.train()
            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(batch["input_ids"][data_idx][token_idx])
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"

                    if text_table is not None:
                        text_table.add_data(total_train_steps, text_str)
                        wandb_run.log({"data_table": copy(text_table)})

                    total_train_steps += 1

                batch = _move_batch_to_device(batch, local_rank)

                # SIM-CoT branch expects explainable_ids_list as a flat List[int]
                # with << >> delimiters marking step boundaries per sample.
                # data.py returns steps_tokenized as List[List[int]] per sample
                # (nested: one inner list per reasoning step, no delimiters).
                # We flatten here and wrap each step with << >> tokens.
                if "steps_tokenized" in batch and "explainable_ids_list" not in batch:
                    raw_steps = batch.pop("steps_tokenized")
                    batch["explainable_ids_list"] = _build_explainable_ids_list(
                        raw_steps=raw_steps,
                        input_ids=batch["input_ids"],
                        latent_id=latent_id,
                        c_thought=configs.c_thought,
                        l_id=l_id,
                        r_id=r_id,
                    )

                outputs = parallel_model(
                    **{k: v for k, v in batch.items() if k != "idx"}
                )
                loss_breakdown = getattr(parallel_model.module, "last_loss_breakdown", None)
                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == (
                    len(train_dataloader) - 1
                ):
                    grad_health = _grad_health_report(parallel_model.module)
                    step_base_grad_norm = None
                    step_explain_grad_norm = None
                    if hasattr(parallel_model.module, "base_causallm"):
                        step_base_grad_norm = _module_grad_norm(parallel_model.module.base_causallm)
                    if hasattr(parallel_model.module, "expainable_llm"):
                        step_explain_grad_norm = _module_grad_norm(parallel_model.module.expainable_llm)

                    grad_norm_value = None
                    if max_grad_norm > 0:
                        if hasattr(parallel_model, "clip_grad_norm_"):
                            grad_norm = parallel_model.clip_grad_norm_(max_grad_norm)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                parallel_model.parameters(),
                                max_grad_norm,
                            )
                        grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                else:
                    grad_norm_value = None
                    grad_health = None
                    step_base_grad_norm = None
                    step_explain_grad_norm = None

                if rank == 0:
                    loss_scalar = loss.detach().item()
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss_scalar * configs.gradient_accumulation_steps,
                    }
                    base_loss_scalar = None
                    explain_loss_scalar = None
                    total_loss_scalar = None
                    base_valid_targets = None
                    base_logits_finite = None
                    c_thought_num_dbg = None
                    explain_valid_targets = None
                    explain_logits_finite = None
                    explain_eff_min = None
                    explain_eff_max = None
                    base_input_embeds_finite = None
                    base_hidden_finite = None
                    grad_total_bad = None
                    grad_bad_items = None
                    grad_max_abs = None
                    base_grad_norm = step_base_grad_norm
                    explain_grad_norm = step_explain_grad_norm
                    if isinstance(loss_breakdown, dict):
                        base_loss_scalar = loss_breakdown.get("base_loss")
                        explain_loss_scalar = loss_breakdown.get("explain_loss")
                        total_loss_scalar = loss_breakdown.get("total_loss")
                        base_valid_targets = loss_breakdown.get("base_valid_targets")
                        base_logits_finite = loss_breakdown.get("base_logits_finite")
                        c_thought_num_dbg = loss_breakdown.get("c_thought_num")
                        explain_valid_targets = loss_breakdown.get("explain_valid_targets")
                        explain_logits_finite = loss_breakdown.get("explain_logits_finite")
                        explain_eff_min = loss_breakdown.get("explain_effective_count_min")
                        explain_eff_max = loss_breakdown.get("explain_effective_count_max")
                        base_input_embeds_finite = loss_breakdown.get("base_input_embeds_finite")
                        base_hidden_finite = loss_breakdown.get("base_hidden_finite")
                        if base_loss_scalar is not None:
                            log_dict["train/base_loss"] = base_loss_scalar
                        if explain_loss_scalar is not None:
                            log_dict["train/explain_loss"] = explain_loss_scalar
                        if total_loss_scalar is not None:
                            log_dict["train/total_loss"] = total_loss_scalar
                        if base_valid_targets is not None:
                            log_dict["train/base_valid_targets"] = base_valid_targets
                        if explain_valid_targets is not None:
                            log_dict["train/explain_valid_targets"] = explain_valid_targets
                    if grad_norm_value is not None:
                        log_dict["train/grad_norm"] = grad_norm_value
                    if (step + 1) % configs.gradient_accumulation_steps == 0 or step == (
                        len(train_dataloader) - 1
                    ):
                        grad_total_bad = grad_health["total_bad"]
                        grad_bad_items = grad_health["bad_items"]
                        grad_max_abs = grad_health["max_abs_grad"]
                        if grad_total_bad is not None:
                            log_dict["train/grad_nonfinite_count"] = grad_total_bad
                        if grad_max_abs is not None:
                            log_dict["train/grad_max_abs"] = grad_max_abs
                        if base_grad_norm is not None and math.isfinite(base_grad_norm):
                            log_dict["train/base_grad_norm"] = base_grad_norm
                        if explain_grad_norm is not None and math.isfinite(explain_grad_norm):
                            log_dict["train/explain_grad_norm"] = explain_grad_norm

                    if wandb_run:
                        wandb_run.log(log_dict)

                    if step % loss_log_interval == 0:
                        print(
                            f"[loss_breakdown] epoch={epoch+1} step={step} "
                            f"total={total_loss_scalar if total_loss_scalar is not None else 'NA'} "
                            f"base={base_loss_scalar if base_loss_scalar is not None else 'NA'} "
                            f"explain={explain_loss_scalar if explain_loss_scalar is not None else 'NA'} "
                            f"| base_targets={base_valid_targets if base_valid_targets is not None else 'NA'} "
                            f"base_logits_finite={base_logits_finite if base_logits_finite is not None else 'NA'} "
                            f"explain_targets={explain_valid_targets if explain_valid_targets is not None else 'NA'} "
                            f"explain_logits_finite={explain_logits_finite if explain_logits_finite is not None else 'NA'} "
                            f"base_input_embeds_finite={base_input_embeds_finite if base_input_embeds_finite is not None else 'NA'} "
                            f"base_hidden_finite={base_hidden_finite if base_hidden_finite is not None else 'NA'} "
                            f"c_thought_num={c_thought_num_dbg if c_thought_num_dbg is not None else 'NA'} "
                            f"explain_eff_min={explain_eff_min if explain_eff_min is not None else 'NA'} "
                            f"explain_eff_max={explain_eff_max if explain_eff_max is not None else 'NA'} "
                            f"grad_norm={grad_norm_value if grad_norm_value is not None else 'NA'}"
                        )
                    if (
                        ((step + 1) % configs.gradient_accumulation_steps == 0 or step == (len(train_dataloader) - 1))
                        and (
                            grad_total_bad is not None
                            and (grad_total_bad > 0 or step % grad_health_interval == 0)
                        )
                    ):
                        print(
                            f"[grad_health] epoch={epoch+1} step={step} "
                            f"nonfinite_grad={grad_total_bad} max_abs_grad={grad_max_abs} "
                            f"base_grad_norm={base_grad_norm} explain_grad_norm={explain_grad_norm} "
                            f"samples={grad_bad_items}"
                        )

                    nan_in_total = (total_loss_scalar is not None) and (not math.isfinite(float(total_loss_scalar)))
                    nan_in_base = (base_loss_scalar is not None) and (not math.isfinite(float(base_loss_scalar)))
                    nan_in_explain = (explain_loss_scalar is not None) and (not math.isfinite(float(explain_loss_scalar)))
                    if nan_in_total or nan_in_base or nan_in_explain:
                        print(
                            f"[nan_debug] epoch={epoch+1} step={step} "
                            f"total={total_loss_scalar} base={base_loss_scalar} explain={explain_loss_scalar} "
                            f"base_targets={base_valid_targets} base_logits_finite={base_logits_finite} "
                            f"explain_targets={explain_valid_targets} explain_logits_finite={explain_logits_finite} "
                            f"base_input_embeds_finite={base_input_embeds_finite} "
                            f"base_hidden_finite={base_hidden_finite} "
                            f"c_thought_num={c_thought_num_dbg} "
                            f"explain_eff_min={explain_eff_min} explain_eff_max={explain_eff_max}"
                        )

                    extra_loss_info = ""
                    if base_loss_scalar is not None:
                        extra_loss_info += f", base: {round(float(base_loss_scalar), 4)}"
                    if explain_loss_scalar is not None:
                        extra_loss_info += f", explain: {round(float(explain_loss_scalar), 4)}"
                    if grad_norm_value is not None:
                        extra_loss_info += f", gnorm: {round(float(grad_norm_value), 4)}"
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{configs.num_epochs}, "
                        f"batch {step}/{len(train_dataloader)} completed "
                        f"(loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}{extra_loss_info})"
                    )

            pbar.close()
            dist.barrier()

            if (not configs.save_only_improve) and (not configs.debug) and (
                not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    print("saving model.")
                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

        # -------------------------
        # VAL LOSS
        # -------------------------
        total_loss = 0.0
        with torch.no_grad():
            parallel_model.module.eval()
            for step, batch in enumerate(valid_loss_dataloader):
                batch = _move_batch_to_device(batch, local_rank)

                if "steps_tokenized" in batch and "explainable_ids_list" not in batch:
                    raw_steps = batch.pop("steps_tokenized")
                    batch["explainable_ids_list"] = _build_explainable_ids_list(
                        raw_steps=raw_steps,
                        input_ids=batch["input_ids"],
                        latent_id=latent_id,
                        c_thought=configs.c_thought,
                        l_id=l_id,
                        r_id=r_id,
                    )

                outputs = parallel_model(
                    **{k: v for k, v in batch.items() if k != "idx"}
                )
                loss = outputs.loss
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                total_loss += loss.item() / world_size

            if rank == 0:
                log_dict = {"eval/loss": total_loss / len(valid_loss_dataloader)}
                if wandb_run:
                    wandb_run.log(log_dict)
                print("eval loss", total_loss / len(valid_loss_dataloader))

        # -------------------------
        # OPTIONAL: GENERATION EVAL
        # -------------------------
        if skip_generation_eval:
            continue

        # If you want generation-based eval, you must provide your own parsing logic
        # for "correctness" of the generated next action. For now we just print examples.
        with torch.no_grad():
            parallel_model.module.eval()
            shown = 0
            for _, batch in enumerate(valid_gen_dataloader):
                # avoid passing position_ids to HF generate in some configs
                batch = {
                    k: v.to(local_rank)
                    for k, v in batch.items()
                    if (v is not None) and (k not in ["idx", "position_ids"])
                }
                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )
                if rank == 0 and shown < 5:
                    print("=== SAMPLE GENERATION ===")
                    print(tokenizer.decode(outputs[0]))
                    shown += 1

        if configs.only_eval:
            break


if __name__ == "__main__":
    main()
