# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import functools
import gc
import json
import os
import sys

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

    set_seed(configs.seed)

    save_dir = os.path.join(configs.save_path, configs.name)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    dist.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.
    if len(cur_ckpts) > 0 and not configs.only_eval:
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

    trust_remote = bool(raw_cfg.get("trust_remote_code", False))

    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        trust_remote_code=trust_remote,
    ).to(local_rank)

    if configs.mode != "coconut_baseline":
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

                # SIM-CoT branch expects explainable_ids_list
                if "steps_tokenized" in batch and "explainable_ids_list" not in batch:
                    batch["explainable_ids_list"] = batch.pop("steps_tokenized")

                outputs = parallel_model(
                    **{k: v for k, v in batch.items() if k != "idx"}
                )
                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == (
                    len(train_dataloader) - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if rank == 0:
                    loss_scalar = loss.detach().item()
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss_scalar * configs.gradient_accumulation_steps,
                    }
                    if wandb_run:
                        wandb_run.log(log_dict)
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{configs.num_epochs}, "
                        f"batch {step}/{len(train_dataloader)} completed "
                        f"(loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)})"
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
                    batch["explainable_ids_list"] = batch.pop("steps_tokenized")

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
