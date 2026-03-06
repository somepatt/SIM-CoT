import json
import itertools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from loguru import logger

PROMPT_TEMPLATE = """You are an autonomous software engineer tasked with fixing bugs.
Analyze the issue and produce a minimal unified diff that resolves the failure.
If you are unsure, still return your best attempt at a patch.

Repository: {repo}
Base commit: {base_commit}
Instance ID: {instance_id}

Problem statement:
{problem}

Provide ONLY the patch in a fenced diff block or raw unified diff.
"""


def _strip_code_block(text: str) -> str:
    if "```" in text:
        return text.split("```", 1)[0].strip()
    return text.strip()


def _looks_like_observation(content: str) -> bool:
    stripped = content.lstrip()
    return stripped.startswith(("<returncode>", "<output>", "<warning>", "<error>"))


def _format_chat_messages(messages: List[Dict]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role") or "user"
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and _looks_like_observation(content):
            label = "Observation"
        else:
            label = role.capitalize()
        parts.append(f"{label}:\n{content}")
    return "\n\n".join(parts).strip()


def _split_assistant_turn(content: str) -> Optional[Dict[str, str]]:
    match = re.search(r"```[^\n]*\n(.*?)```", content, flags=re.DOTALL)
    if not match:
        return None
    thought_text = content[:match.start()].strip()
    if thought_text.startswith("THOUGHT:"):
        thought_text = thought_text[len("THOUGHT:"):].strip()
    code_text = match.group(1).strip()
    if not code_text:
        return None
    return {"thought": thought_text, "code": code_text}


def _extract_steps_from_messages(messages: List[Dict]) -> List[str]:
    steps: List[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        split = _split_assistant_turn(content)
        if split and split["thought"]:
            steps.append(split["thought"])
            continue
        step_text = _strip_code_block(content)
        if step_text:
            steps.append(step_text)
    return steps


def _extract_steps(
    sample: Dict,
    steps_field: Optional[str],
    steps_pattern: Optional[str],
    steps_delimiter: Optional[str],
) -> List[str]:
    if steps_field and steps_field in sample:
        raw_steps = sample[steps_field]
        if isinstance(raw_steps, list):
            steps = raw_steps
        elif isinstance(raw_steps, str):
            if steps_pattern:
                steps = re.findall(steps_pattern, raw_steps, flags=re.DOTALL)
            elif steps_delimiter:
                steps = raw_steps.split(steps_delimiter)
            else:
                steps = [raw_steps]
        else:
            steps = []
    elif "messages" in sample:
        steps = _extract_steps_from_messages(sample["messages"])
    else:
        steps = []

    cleaned: List[str] = []
    for step in steps:
        if isinstance(step, str):
            text = step.strip()
            if text:
                cleaned.append(text)
    return cleaned


def _load_trajectory_dataset(dataset_path: str) -> Dataset:
    """
    Загружает траектории для SWE-агента.
    Для каждого шага создает sample с инкрементальным контекстом:
      - prompt: все сообщения до текущего assistant (включая observations)
      - steps: [thought] - мысль на текущем шаге
      - patch: code - команда для выполнения
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory dataset path not found: {dataset_path}")

    files = sorted(path.glob("*.traj.json"))
    if not files:
        raise FileNotFoundError(f"No .traj.json files found in: {dataset_path}")

    samples: List[Dict] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Skipping {file} due to JSON error: {exc}")
            continue

        messages = data.get("messages", [])
        instance_id = data.get("instance_id")

        # Инкрементально накапливаем контекст
        accumulated_context: List[Dict] = []
        accumulated_thoughts: List[str] = []
        step_counter = 0

        for idx, message in enumerate(messages):
            # Добавляем текущее сообщение в accumulated_context ПОСЛЕ обработки
            if message.get("role") == "assistant":
                content = message.get("content") or ""
                split = _split_assistant_turn(content)

                if split and split["code"]:
                    # Создаем prompt из всего накопленного контекста
                    prompt = _format_chat_messages(accumulated_context)
                    thought = split["thought"].strip()
                    code = split["code"]

                    if prompt and code:
                        sample_id = f"{instance_id}-step{step_counter}" if instance_id else None
                        current_steps = list(accumulated_thoughts)
                        if thought:
                            current_steps.append(thought)
                        samples.append(
                            {
                                "instance_id": sample_id,
                                "prompt": prompt,
                                "patch": code,
                                "steps": current_steps,
                            }
                        )
                        step_counter += 1
                        if thought:
                            accumulated_thoughts.append(thought)

            # Добавляем сообщение в контекст для следующих шагов
            accumulated_context.append(message)

    if not samples:
        raise ValueError(f"No usable samples found in {dataset_path}")

    logger.info(f"Loaded {len(samples)} trajectory samples from {len(files)} files")
    return Dataset.from_list(samples)

def build_train_prompt(sample: Dict) -> str:
    """Строим ровно тот же промпт, что и при инференсе."""
    repo = sample.get("repo") or sample.get("repo_name") or "unknown repo"
    base_commit = sample.get("base_commit") or sample.get("base_commit_id") or "unknown"
    problem = sample.get("problem_statement") or sample.get("prompt") or ""
    instance_id = sample.get("instance_id") or sample.get("id") or "NA"

    return PROMPT_TEMPLATE.format(
        repo=repo,
        base_commit=base_commit,
        instance_id=instance_id,
        problem=problem.strip(),
    ).strip() + "\n"


def get_dataset(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_size: int = 20000,
    max_seq_length: int = 8192,
    steps_field: Optional[str] = None,
    steps_pattern: Optional[str] = None,
    steps_delimiter: Optional[str] = None,
    max_step_tokens: int = 256,
    require_steps: bool = False,
) -> Dataset:
    """
    Загружаем датасет и превращаем его в:
      - question_tokenized: prompt (репо + проблема)
      - steps_tokenized: шаги рассуждений (если доступны)
      - answer_tokenized: ПОЛНЫЙ patch (diff) + EOS

    Модель будет учиться предсказывать именно patch.
    """
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        logger.info(f"Loading local trajectory dataset from '{dataset_name}'...")
        dataset = _load_trajectory_dataset(dataset_name)
    else:
        logger.info(f"Loading dataset '{dataset_name}' split '{split}' from Hugging Face...")

        try:
            dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.error(f"Не удалось загрузить датасет '{dataset_name}': {e}")
            raise

    if max_size < len(dataset):
        logger.warning(f"Using a subset of {max_size} samples (full size: {len(dataset)})")
        dataset = dataset.select(range(max_size))

    def tokenize_sample(sample, idx):
        """
        Tokenizes a sample для SIM-CoT обучения:
          - question_tokenized: промпт (repo + проблема ИЛИ траектория контекст)
          - steps_tokenized: шаги рассуждений (thoughts) для SIM-CoT decoder
          - answer_tokenized: команда/patch для выполнения + EOS
        """
        if "prompt" in sample and "repo" not in sample and "problem_statement" not in sample:
            prompt = (sample.get("prompt") or "").strip()
        else:
            prompt = build_train_prompt(sample)

        question_tokenized = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        patch_text = sample.get("patch", "")

        answer_tokenized = tokenizer.encode(
            patch_text + tokenizer.eos_token,
            add_special_tokens=False,
        )

        # Извлекаем steps (thoughts) для SIM-CoT
        steps_text = _extract_steps(sample, steps_field, steps_pattern, steps_delimiter)
        steps_tokenized = [
            tokenizer.encode(step, add_special_tokens=False)[:max_step_tokens]
            for step in steps_text
        ]

        # ✅ ДОБАВЛЕНО: Валидация токенов - убираем invalid token IDs
        vocab_size = len(tokenizer)
        steps_tokenized = [
            [t for t in tokens if 0 <= t < vocab_size]
            for tokens in steps_tokenized
        ]
        # Убираем пустые steps
        steps_tokenized = [tokens for tokens in steps_tokenized if tokens]

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": idx,
        }

    dataset = dataset.map(
        tokenize_sample,
        with_indices=True,
        remove_columns=list(dataset.features),
        num_proc=32,
        desc="Tokenizing",
    )

    prev_len = len(dataset)

    def filter_long_samples(sample):
        patch_len = len(sample["answer_tokenized"])
        total_len = len(sample["question_tokenized"]) + patch_len

        if patch_len == 0:
            return False

        if total_len > max_seq_length:
            return False

        if require_steps and not sample["steps_tokenized"]:
            return False

        return True

    dataset = dataset.filter(
        filter_long_samples,
        num_proc=32,
        desc="Filtering by length",
    )

    logger.info(f"Filtered {prev_len - len(dataset)} samples exceeding {max_seq_length} tokens.")
    logger.info(f"Dataset loaded and filtered: {len(dataset)} samples")

    return dataset


@dataclass
class MyCollator:
    """Collate with KV cache optimization"""
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        if not features:
            return {}

        assert self.tokenizer.padding_side == "right"

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)

            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0

                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )

                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]

                if "labels" in feature:
                    feature["labels"] = [
                        self.label_pad_token_id
                    ] * n_tok_pad + feature["labels"]

                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"
        steps_key = "steps_tokenized"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids" and k != steps_key
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = torch.tensor(
                [
                    label + [self.label_pad_token_id] * (max_label_length - len(label))
                    for label in labels
                ],
                dtype=torch.int64,
            )

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = torch.tensor(
                [
                    [0] * (max_pos_length - len(position_id)) + position_id
                    for position_id in position_ids
                ],
                dtype=torch.int64,
            )

        if steps_key in features[0]:
            batch[steps_key] = [feature.get(steps_key, []) for feature in features]

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker: bool = False,
):
    """
    Dataset для генерации (evaluation).

    Так как CoT нет, количество latent-токенов определяется только:
      - current stage (scheduled_stage)
      - max_latent_stage
      - c_thought

    k = min(stage, max_latent_stage) * c_thought
    """

    max_latent_stage = configs.get("max_latent_stage", 5)
    c_thought = configs.get("c_thought", 1)

    def process_dataset(sample):
        available_steps = len(sample.get("steps_tokenized", []))
        k = min(scheduled_stage, max_latent_stage, available_steps) * c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        output = {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
        if "steps_tokenized" in sample:
            output["steps_tokenized"] = sample["steps_tokenized"]
        return output

    remove_columns = [
        col for col in base_dataset.features if col != "steps_tokenized"
    ]
    return base_dataset.map(
        process_dataset,
        remove_columns=remove_columns,
        num_proc=32,
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker: bool = False,
    shuffle: bool = False,
):
    """
    ✅ ИСПРАВЛЕНО: Dataset для обучения с latent-токенами для IMPLICIT CoT.

    Использует латентные токены (<thought>) вместо явного text-based reasoning.
    Steps (thoughts) используются только для SIM-CoT decoder supervision,
    НЕ включаются в основную последовательность.

    Структура:
      - Вход: question + <bot> + <thought>*N + <eot> + patch + eos
      - Labels: mask(prompt + bot/thought/eot) + patch+eos
      - steps_tokenized: передаются отдельно для SIM-CoT decoder

    Количество latent-токенов N растёт с каждым stage:
      N = min(stage, max_latent_stage) * c_thought
    """

    n_additional_tokens = 0 if no_special_marker else 2

    max_seq_len = configs.get("max_seq_length", 8192)
    max_latent_stage = configs.get("max_latent_stage", 5)
    c_thought = configs.get("c_thought", 1)
    no_cot_flag = configs.get("no_cot", False)

    def process_dataset(sample):
        if no_cot_flag:
            n_latent_tokens = 0
        else:
            available_steps = len(sample.get("steps_tokenized", []))
            stage_clamped = min(scheduled_stage, max_latent_stage, available_steps)
            n_latent_tokens = stage_clamped * c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + sample["answer_tokenized"]
        )

        prefix_len = (
            len(sample["question_tokenized"])
            + n_latent_tokens
            + n_additional_tokens
        )
        labels = [-100] * prefix_len + tokens[prefix_len:]

        tokens = tokens[:max_seq_len]
        labels = labels[:max_seq_len]

        # # DEBUG: print only once to avoid log spam
        # if sample["idx"] == 0:
        #     print("-" * 60)
        #     print("DATASET DEBUG process_dataset idx=0")
        #     print("latentid arg:", latent_id)
        #     print("startid arg:", start_id, "endid arg:", end_id)
        #     print("nlatenttokens:", n_latent_tokens)
        #     print("nocotflag", no_cot_flag)
        #     print("stage_clamped", stage_clamped, "c_thought", c_thought)
        #     print("latentid in tokens?:", latent_id in tokens)
        #     if latent_id in tokens:
        #         first = tokens.index(latent_id)
        #         print("first latent pos:", first)
        #         print("window:", tokens[max(0, first-5):first+10])
        #     else:
        #         print("NO latent token found in constructed tokens")
        #     print("-" * 60)


        output = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }
        if "steps_tokenized" in sample:
            output["steps_tokenized"] = sample["steps_tokenized"]
        return output

    remove_columns = [
        col for col in base_dataset.features if col != "steps_tokenized"
    ]
    processed_dataset = base_dataset.map(
        process_dataset,
        remove_columns=remove_columns,
        num_proc=32,
    )

    if shuffle:
        processed_dataset = processed_dataset.shuffle()

    return processed_dataset
