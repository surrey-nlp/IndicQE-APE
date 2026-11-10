"""MQM task runner: prompt → generate → parse error spans → severity-weighted score."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..cache import ResponseCache, fewshot_signature, prompt_hash
from ..data import DataLoader
from ..languages import parse_pair
from ..metrics import mqm_dataset_score
from ..models import LLMProvider
from ..models.base import GenerationConfig
from ..prompts import Example, parse_mqm_response, render_mqm
from .qe import _read_done_ids


def run_mqm(
    provider: LLMProvider,
    *,
    pair: str = "enhi",
    subset: str = "mqm_en_hi",
    split: str = "test",
    shots: int = 0,
    limit: int | None = None,
    seed: int = 0,
    out: str | Path,
    cache_dir: str | Path = "cache",
    data_loader: DataLoader | None = None,
) -> dict:
    """Generate MQM error spans and score them against the gold annotations."""
    pair_obj = parse_pair(pair)
    dl = data_loader or DataLoader()
    cache = ResponseCache(Path(cache_dir) / f"mqm__{provider.name}__{provider.model.replace('/', '__')}.jsonl")
    cfg = GenerationConfig(max_tokens=512, temperature=0.0)

    fewshot_examples: list[Example] = []
    if shots > 0:
        for ex in dl.iter_items(subset, pair=pair, limit=shots, seed=seed + 10_000):
            if ex.mqm_spans is None:
                continue
            label = "\n".join(
                f"{e.get('severity','minor')} | {e.get('category','mistranslation')} | {e.get('span','')}"
                for e in ex.mqm_spans
            )
            fewshot_examples.append(Example(source=ex.source, mt=ex.mt, label=label))

    fs_sig = fewshot_signature(fewshot_examples)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _read_done_ids(out_path)
    is_resume = bool(done_ids)
    open_mode = "a" if is_resume else "w"

    with out_path.open(open_mode, encoding="utf-8") as fh:
        if not is_resume:
            fh.write(json.dumps({
                "_meta": {
                    "task": "mqm", "subset": subset, "pair": pair, "shots": shots,
                    "provider": provider.name, "model": provider.model,
                    "limit": limit, "seed": seed,
                    "fewshot_signature": fs_sig,
                    "generation": asdict(cfg),
                }
            }) + "\n")

        items = list(dl.iter_items(subset, pair=pair, limit=limit, seed=seed))
        for it in tqdm(items, desc=f"mqm/{pair}/{provider.name}:{provider.model}"):
            if it.id in done_ids:
                continue
            sys_msg, usr_msg = render_mqm(pair_obj, it.source, it.mt, fewshot=fewshot_examples)
            key = prompt_hash(
                provider.name, provider.model, sys_msg, usr_msg,
                task="mqm", shots=shots, fewshot_signature=fs_sig,
                **asdict(cfg),
            )
            response = cache.get(key)
            if response is None:
                response = provider.complete(sys_msg, usr_msg, config=cfg)
                cache.put(key, response)
            pred = parse_mqm_response(response)
            gold = list(it.mqm_spans or [])
            fh.write(json.dumps({
                "id": it.id, "pair": it.pair, "task": "mqm",
                "gold_spans": gold, "pred_spans": pred,
                "raw_response": response,
            }, ensure_ascii=False) + "\n")
            fh.flush()

    # Recompute aggregate metrics from the full output file.
    pred_errors: list[list[dict]] = []
    gold_errors: list[list[dict]] = []
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in rec or "_summary" in rec:
                continue
            pred_errors.append(rec.get("pred_spans", []))
            gold_errors.append(rec.get("gold_spans", []))

    summary = mqm_dataset_score(pred_errors, gold_errors)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_summary": summary}) + "\n")
    return summary
