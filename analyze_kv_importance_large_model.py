#!/usr/bin/env python3
"""Run KV importance / hot-cold stability analysis with a manual decode loop."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_PREFIX = "Context:\n"
PROMPT_BETWEEN = "\n\nQuery:\n"
PROMPT_SUFFIX = "\n\nAnswer:\n"


@dataclass
class Sample:
    sample_id: str
    context: str
    query: str
    request_type: str


@dataclass
class TokenizedPrompt:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    original_tokens: int
    truncated_tokens: int
    was_truncated: bool
    strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token/chunk KV importance from attention weights."
    )
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face model name")
    parser.add_argument("--input", default="data/samples.jsonl", help="Input JSONL file")
    parser.add_argument(
        "--output-dir",
        default="outputs/kv_importance",
        help="Directory for CSV and PNG outputs",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--sink-size", type=int, default=8)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional debug helper: only process the first N samples.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warn("CUDA was requested but is not available; falling back to CPU.")
    return torch.device("cpu")


def read_samples(path: Path, limit: Optional[int] = None) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            missing = {"id", "context", "query", "type"} - set(row)
            if missing:
                raise ValueError(
                    f"Sample on line {line_number} is missing fields: {sorted(missing)}"
                )

            request_type = str(row["type"])
            if request_type not in {"local", "long_range"}:
                warn(
                    f"Sample {row['id']} has unknown type {request_type!r}; "
                    "using local truncation strategy."
                )
                request_type = "local"

            samples.append(
                Sample(
                    sample_id=str(row["id"]),
                    context=str(row["context"]),
                    query=str(row["query"]),
                    request_type=request_type,
                )
            )
            if limit is not None and len(samples) >= limit:
                break

    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def tokenizer_encode(tokenizer: Any, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def truncate_context_ids(
    context_ids: Sequence[int], budget: int, request_type: str
) -> Tuple[List[int], str]:
    if budget <= 0:
        return [], f"{request_type}:context_dropped"
    if len(context_ids) <= budget:
        return list(context_ids), f"{request_type}:not_truncated"

    if request_type == "long_range":
        prefix_len = max(1, int(round(budget * 0.45)))
        middle_len = max(0, int(round(budget * 0.45)))
        suffix_len = budget - prefix_len - middle_len
        if suffix_len < 0:
            middle_len += suffix_len
            suffix_len = 0

        context_len = len(context_ids)
        middle_start = max(prefix_len, (context_len - middle_len) // 2)
        middle_end = min(context_len - suffix_len, middle_start + middle_len)
        middle_start = max(prefix_len, middle_end - middle_len)

        selected = (
            list(context_ids[:prefix_len])
            + list(context_ids[middle_start:middle_end])
            + (list(context_ids[-suffix_len:]) if suffix_len else [])
        )
        return selected[:budget], "long_range:preserve_prefix_middle_tail"

    return list(context_ids[-budget:]), "local:preserve_recent_context"


def build_prompt_input(
    tokenizer: Any,
    sample: Sample,
    max_input_tokens: int,
    device: torch.device,
) -> TokenizedPrompt:
    if max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be at least 1")

    prefix_ids = tokenizer_encode(tokenizer, PROMPT_PREFIX)
    between_ids = tokenizer_encode(tokenizer, PROMPT_BETWEEN)
    suffix_ids = tokenizer_encode(tokenizer, PROMPT_SUFFIX)
    context_ids = tokenizer_encode(tokenizer, sample.context)
    query_ids = tokenizer_encode(tokenizer, sample.query)

    original_ids = prefix_ids + context_ids + between_ids + query_ids + suffix_ids
    original_tokens = len(original_ids)
    if original_tokens <= max_input_tokens:
        input_ids = original_ids
        strategy = "not_truncated"
        was_truncated = False
    else:
        fixed_overhead = len(prefix_ids) + len(between_ids) + len(suffix_ids)
        max_query_tokens = max_input_tokens - fixed_overhead
        if max_query_tokens <= 0:
            raise ValueError(
                "--max-input-tokens is too small for the prompt template; "
                f"need more than {fixed_overhead} tokens."
            )

        query_strategy = "query_full"
        if len(query_ids) > max_query_tokens:
            query_ids = list(query_ids[-max_query_tokens:])
            context_ids = []
            query_strategy = "query_truncated_preserve_tail"
        else:
            context_budget = max_input_tokens - fixed_overhead - len(query_ids)
            context_ids, context_strategy = truncate_context_ids(
                context_ids, context_budget, sample.request_type
            )
            query_strategy = context_strategy

        input_ids = prefix_ids + list(context_ids) + between_ids + list(query_ids) + suffix_ids
        strategy = query_strategy
        was_truncated = True
        warn(
            f"Sample {sample.sample_id} truncated from {original_tokens} to "
            f"{len(input_ids)} tokens using {strategy}."
        )

    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
        was_truncated = True
        strategy = f"{strategy}:safety_tail_trim"

    if not input_ids:
        raise ValueError(f"Sample {sample.sample_id} produced an empty prompt.")

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor, device=device)
    return TokenizedPrompt(
        input_ids=tensor,
        attention_mask=attention_mask,
        original_tokens=original_tokens,
        truncated_tokens=tensor.shape[1],
        was_truncated=was_truncated,
        strategy=strategy,
    )


def infer_model_context_limit(model: Any, tokenizer: Any) -> Optional[int]:
    candidates: List[int] = []
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        candidates.append(tokenizer_limit)

    if not candidates:
        return None
    return min(candidates)


def effective_max_input_tokens(
    requested_max_input_tokens: int,
    max_new_tokens: int,
    model_context_limit: Optional[int],
) -> int:
    if model_context_limit is None:
        return requested_max_input_tokens

    max_allowed = model_context_limit - max_new_tokens
    if max_allowed < 1:
        raise ValueError(
            f"Model context limit ({model_context_limit}) is too small for "
            f"--max-new-tokens {max_new_tokens}."
        )

    if requested_max_input_tokens > max_allowed:
        warn(
            f"--max-input-tokens {requested_max_input_tokens} plus "
            f"--max-new-tokens {max_new_tokens} exceeds model context limit "
            f"{model_context_limit}; using effective max input tokens {max_allowed}."
        )
        return max_allowed
    return requested_max_input_tokens


def load_model_and_tokenizer(
    model_name: str, device: torch.device
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_kwargs: Dict[str, Any] = {}
    if device.type == "cuda":
        dtype_kwargs["torch_dtype"] = torch.float16

    load_kwargs: Dict[str, Any] = {
        **dtype_kwargs,
        "output_attentions": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError as exc:
        warn(
            "Model loader does not accept attn_implementation='eager'; "
            f"retrying without it ({exc})."
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except ValueError as exc:
        warn(
            "Model does not support attn_implementation='eager'; "
            f"retrying without it ({exc})."
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model.config, "output_attentions"):
        model.config.output_attentions = True
    return model, tokenizer


def extract_token_importance(attentions: Any) -> np.ndarray:
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Try a model/configuration that supports "
            "output_attentions=True, or an eager attention implementation."
        )

    layer_vectors: List[torch.Tensor] = []
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        if layer_attention.ndim != 4:
            raise RuntimeError(
                "Expected attention shape [batch, heads, query_len, key_len], "
                f"got {tuple(layer_attention.shape)}"
            )
        current_query_attention = layer_attention[:, :, -1, :]
        head_mean = current_query_attention.float().mean(dim=(0, 1))
        layer_vectors.append(head_mean)

    if not layer_vectors:
        raise RuntimeError("No layer attention tensors were returned by the model.")

    token_importance = torch.stack(layer_vectors, dim=0).mean(dim=0)
    token_importance = torch.nan_to_num(token_importance, nan=0.0, posinf=0.0, neginf=0.0)
    total = token_importance.sum()
    if total.item() > 0:
        token_importance = token_importance / total
    return token_importance.detach().cpu().numpy()


def chunk_importance_from_tokens(
    token_importance: np.ndarray, chunk_size: int
) -> List[Dict[str, Any]]:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    rows: List[Dict[str, Any]] = []
    total_len = len(token_importance)
    num_chunks = int(math.ceil(total_len / chunk_size))
    for chunk_id in range(num_chunks):
        token_start = chunk_id * chunk_size
        token_stop = min((chunk_id + 1) * chunk_size, total_len)
        rows.append(
            {
                "chunk_id": chunk_id,
                "token_start": token_start,
                "token_end": token_stop - 1,
                "importance": float(token_importance[token_start:token_stop].sum()),
            }
        )
    return rows


def attention_mass(
    token_importance: np.ndarray, sink_size: int, recent_window: int
) -> Tuple[float, float, float]:
    if sink_size < 0:
        raise ValueError("--sink-size must be non-negative")
    if recent_window < 0:
        raise ValueError("--recent-window must be non-negative")

    total_len = len(token_importance)
    sink_end = min(sink_size, total_len)
    recent_start = max(sink_end, total_len - recent_window)

    sink_mass = float(token_importance[:sink_end].sum())
    recent_mass = float(token_importance[recent_start:].sum())
    cold_mass = float(token_importance[sink_end:recent_start].sum())
    return sink_mass, recent_mass, cold_mass


def topk_chunks(
    chunk_rows: Sequence[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    sorted_rows = sorted(
        chunk_rows,
        key=lambda row: (-float(row["importance"]), int(row["chunk_id"])),
    )
    return [
        {
            "topk_rank": rank,
            "chunk_id": int(row["chunk_id"]),
            "importance": float(row["importance"]),
        }
        for rank, row in enumerate(sorted_rows[: max(0, top_k)], start=1)
    ]


def safe_spearman(previous: Dict[int, float], current: Dict[int, float]) -> float:
    all_chunk_ids = sorted(set(previous) | set(current))
    if len(all_chunk_ids) < 2:
        return float("nan")
    prev_values = np.array([previous.get(chunk_id, 0.0) for chunk_id in all_chunk_ids])
    curr_values = np.array([current.get(chunk_id, 0.0) for chunk_id in all_chunk_ids])
    if np.allclose(prev_values, prev_values[0]) or np.allclose(curr_values, curr_values[0]):
        return float("nan")
    corr = spearmanr(prev_values, curr_values).correlation
    return float(corr) if corr is not None else float("nan")


def jaccard(previous: Iterable[int], current: Iterable[int]) -> float:
    previous_set = set(previous)
    current_set = set(current)
    union = previous_set | current_set
    if not union:
        return float("nan")
    return len(previous_set & current_set) / len(union)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def run_sample(
    model: Any,
    tokenizer: Any,
    sample: Sample,
    tokenized: TokenizedPrompt,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    chunk_rows_all: List[Dict[str, Any]] = []
    topk_rows_all: List[Dict[str, Any]] = []
    stability_rows_all: List[Dict[str, Any]] = []
    mass_rows_all: List[Dict[str, Any]] = []
    generated_token_ids: List[int] = []

    with torch.inference_mode():
        # Prefill attention is not returned because this experiment analyzes
        # decode-time attention only. Full-prompt KV cache construction is unchanged.
        prefill_outputs = model(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = prefill_outputs.past_key_values
        current_token = greedy_next_token(prefill_outputs.logits)
        del prefill_outputs

        previous_topk_set: Optional[set[int]] = None
        previous_chunk_importance: Optional[Dict[int, float]] = None

        for step in range(args.max_new_tokens):
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = outputs.past_key_values
            token_importance = extract_token_importance(outputs.attentions)

            chunk_rows = chunk_importance_from_tokens(token_importance, args.chunk_size)
            chunk_importance_map = {
                int(row["chunk_id"]): float(row["importance"]) for row in chunk_rows
            }

            for row in chunk_rows:
                chunk_rows_all.append(
                    {
                        "sample_id": sample.sample_id,
                        "step": step,
                        **row,
                    }
                )

            step_topk = topk_chunks(chunk_rows, args.top_k)
            current_topk_set = {int(row["chunk_id"]) for row in step_topk}
            for row in step_topk:
                topk_rows_all.append(
                    {
                        "sample_id": sample.sample_id,
                        "step": step,
                        **row,
                    }
                )

            if previous_topk_set is None or previous_chunk_importance is None:
                overlap = float("nan")
                churn = float("nan")
                spearman_corr = float("nan")
            else:
                overlap = jaccard(previous_topk_set, current_topk_set)
                churn = 1.0 - overlap if not math.isnan(overlap) else float("nan")
                spearman_corr = safe_spearman(
                    previous_chunk_importance, chunk_importance_map
                )

            stability_rows_all.append(
                {
                    "sample_id": sample.sample_id,
                    "step": step,
                    "hot_set_overlap": overlap,
                    "churn_rate": churn,
                    "spearman_rank_corr": spearman_corr,
                }
            )
            previous_topk_set = current_topk_set
            previous_chunk_importance = chunk_importance_map

            sink_mass, recent_mass, cold_mass = attention_mass(
                token_importance, args.sink_size, args.recent_window
            )
            mass_rows_all.append(
                {
                    "sample_id": sample.sample_id,
                    "step": step,
                    "sink_mass": sink_mass,
                    "recent_mass": recent_mass,
                    "cold_mass": cold_mass,
                }
            )

            generated_token_ids.append(int(current_token[0, 0].detach().cpu().item()))
            current_token = greedy_next_token(outputs.logits)
            del outputs

            eos_id = tokenizer.eos_token_id
            if eos_id is not None and generated_token_ids[-1] == eos_id:
                break

        del past_key_values
        del current_token

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return (
        chunk_rows_all,
        topk_rows_all,
        stability_rows_all,
        mass_rows_all,
        generated_text,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_chunk_heatmap(chunk_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    if chunk_df.empty:
        plt.text(0.5, 0.5, "No chunk importance data", ha="center", va="center")
        plt.axis("off")
    else:
        mean_df = (
            chunk_df.groupby(["chunk_id", "step"], as_index=False)["importance"]
            .mean()
            .pivot(index="chunk_id", columns="step", values="importance")
            .fillna(0.0)
            .sort_index()
        )
        image = plt.imshow(mean_df.values, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(image, label="mean chunk importance")
        plt.xticks(
            ticks=np.arange(len(mean_df.columns)),
            labels=[str(col) for col in mean_df.columns],
        )
        plt.yticks(
            ticks=np.arange(len(mean_df.index)),
            labels=[str(idx) for idx in mean_df.index],
        )
        plt.xlabel("decode step")
        plt.ylabel("chunk_id")
        plt.title("Chunk Importance Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_hot_set_overlap_plot(stability_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    if stability_df.empty:
        plt.text(0.5, 0.5, "No stability data", ha="center", va="center")
        plt.axis("off")
    else:
        for sample_id, group in stability_df.groupby("sample_id"):
            plt.plot(
                group["step"],
                group["hot_set_overlap"],
                marker="o",
                linewidth=1.2,
                alpha=0.45,
                label=str(sample_id),
            )
        mean_df = stability_df.groupby("step", as_index=False)["hot_set_overlap"].mean()
        plt.plot(
            mean_df["step"],
            mean_df["hot_set_overlap"],
            marker="o",
            linewidth=2.5,
            color="black",
            label="mean",
        )
        plt.ylim(-0.05, 1.05)
        plt.xlabel("decode step")
        plt.ylabel("hot_set_overlap")
        plt.title("Hot Set Overlap")
        plt.legend(fontsize=8, loc="best")
        plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_attention_mass_plot(mass_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    if mass_df.empty:
        plt.text(0.5, 0.5, "No attention mass data", ha="center", va="center")
        plt.axis("off")
    else:
        mean_df = mass_df.groupby("step", as_index=False)[
            ["sink_mass", "recent_mass", "cold_mass"]
        ].mean()
        plt.plot(mean_df["step"], mean_df["sink_mass"], marker="o", label="Sink KV")
        plt.plot(mean_df["step"], mean_df["recent_mass"], marker="o", label="Recent KV")
        plt.plot(mean_df["step"], mean_df["cold_mass"], marker="o", label="Cold KV")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("decode step")
        plt.ylabel("attention mass")
        plt.title("Attention Mass Distribution")
        plt.legend()
        plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_plots(output_dir: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    chunk_df = pd.DataFrame(rows["chunk_importance"])
    stability_df = pd.DataFrame(rows["stability_metrics"])
    mass_df = pd.DataFrame(rows["attention_mass"])

    save_chunk_heatmap(chunk_df, output_dir / "chunk_importance_heatmap.png")
    save_hot_set_overlap_plot(stability_df, output_dir / "hot_set_overlap.png")
    save_attention_mass_plot(mass_df, output_dir / "attention_mass_distribution.png")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading model {args.model_name!r} on {device}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    model_context_limit = infer_model_context_limit(model, tokenizer)
    effective_max_tokens = effective_max_input_tokens(
        args.max_input_tokens, args.max_new_tokens, model_context_limit
    )

    samples = read_samples(Path(args.input), args.limit_samples)
    print(f"Loaded {len(samples)} sample(s).", flush=True)

    all_rows: Dict[str, List[Dict[str, Any]]] = {
        "chunk_importance": [],
        "topk_chunks": [],
        "stability_metrics": [],
        "attention_mass": [],
    }
    truncation_rows: List[Dict[str, Any]] = []
    generated_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, start=1):
        print(
            f"[{sample_index}/{len(samples)}] Running sample {sample.sample_id} "
            f"({sample.request_type})...",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        tokenized = build_prompt_input(tokenizer, sample, effective_max_tokens, device)
        truncation_rows.append(
            {
                "sample_id": sample.sample_id,
                "type": sample.request_type,
                "original_tokens": tokenized.original_tokens,
                "truncated_tokens": tokenized.truncated_tokens,
                "max_input_tokens": effective_max_tokens,
                "was_truncated": tokenized.was_truncated,
                "strategy": tokenized.strategy,
            }
        )

        (
            chunk_rows,
            topk_rows,
            stability_rows,
            mass_rows,
            generated_text,
        ) = run_sample(model, tokenizer, sample, tokenized, args)

        all_rows["chunk_importance"].extend(chunk_rows)
        all_rows["topk_chunks"].extend(topk_rows)
        all_rows["stability_metrics"].extend(stability_rows)
        all_rows["attention_mass"].extend(mass_rows)
        generated_rows.append(
            {
                "sample_id": sample.sample_id,
                "generated_text": generated_text,
            }
        )

        del tokenized
        del chunk_rows, topk_rows, stability_rows, mass_rows, generated_text
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        else:
            allocated_bytes = 0
            reserved_bytes = 0
            peak_allocated_bytes = 0
            peak_reserved_bytes = 0

        memory_rows.append(
            {
                "sample_id": sample.sample_id,
                "allocated_bytes_after_cleanup": allocated_bytes,
                "reserved_bytes_after_cleanup": reserved_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
            }
        )
        print(
            f"Memory after {sample.sample_id}: "
            f"allocated={allocated_bytes / (1024 ** 2):.2f} MiB, "
            f"reserved={reserved_bytes / (1024 ** 2):.2f} MiB, "
            f"peak_allocated={peak_allocated_bytes / (1024 ** 2):.2f} MiB, "
            f"peak_reserved={peak_reserved_bytes / (1024 ** 2):.2f} MiB",
            flush=True,
        )

    write_csv(
        output_dir / "chunk_importance.csv",
        all_rows["chunk_importance"],
        ["sample_id", "step", "chunk_id", "token_start", "token_end", "importance"],
    )
    write_csv(
        output_dir / "topk_chunks.csv",
        all_rows["topk_chunks"],
        ["sample_id", "step", "topk_rank", "chunk_id", "importance"],
    )
    write_csv(
        output_dir / "stability_metrics.csv",
        all_rows["stability_metrics"],
        ["sample_id", "step", "hot_set_overlap", "churn_rate", "spearman_rank_corr"],
    )
    write_csv(
        output_dir / "attention_mass.csv",
        all_rows["attention_mass"],
        ["sample_id", "step", "sink_mass", "recent_mass", "cold_mass"],
    )
    write_csv(
        output_dir / "truncation_log.csv",
        truncation_rows,
        [
            "sample_id",
            "type",
            "original_tokens",
            "truncated_tokens",
            "max_input_tokens",
            "was_truncated",
            "strategy",
        ],
    )
    write_csv(
        output_dir / "generated_text.csv",
        generated_rows,
        ["sample_id", "generated_text"],
    )
    write_csv(
        output_dir / "memory_usage.csv",
        memory_rows,
        [
            "sample_id",
            "allocated_bytes_after_cleanup",
            "reserved_bytes_after_cleanup",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ],
    )
    save_plots(output_dir, all_rows)

    print(f"Done. Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
