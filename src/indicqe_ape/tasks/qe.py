"""QE task runner: prompt → generate → parse → correlate.

Resumable: if `--out` already exists, items already scored (by `id`) are
skipped and the run continues. The summary line is recomputed at the end.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..cache import ResponseCache, fewshot_signature, prompt_hash
from ..data import DataLoader, Item
from ..languages import parse_pair
from ..metrics import correlation, prompt_sensitivity
from ..models import LLMProvider
from ..models.base import GenerationConfig
from ..prompts import DEFAULT_QE_TEMPLATES, QE_TEMPLATES, Example, parse_qe_score, render_qe


def _read_done_ids(path: Path) -> set[str]:
    """Return the set of `id`s already present in an existing JSONL output."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                done.add(str(rec["id"]))
    return done


def run_qe(
    provider: LLMProvider,
    *,
    pair: str,
    subset: str = "challenge",
    split: str = "test",
    shots: int = 0,
    limit: int | None = None,
    seed: int = 0,
    out: str | Path,
    cache_dir: str | Path = "cache",
    templates: tuple[str, ...] | list[str] = DEFAULT_QE_TEMPLATES,
    data_loader: DataLoader | None = None,
    gold: str = "da",
) -> dict:
    """Score `subset`/`split` with `provider` under EACH prompt template, write JSONL, return metrics.

    Every item is scored once per template (default GEMBA-DA + GEMBA-SQM) in a
    single model load. Each record carries `pred`/`raw` keyed by template. The
    summary reports per-template correlations plus a `prompt_sensitivity` block
    comparing the two variants; flat `pearson`/`spearman`/... mirror the primary
    (first) template for backward-compatible readers.
    """
    templates = tuple(templates)
    unknown = [t for t in templates if t not in QE_TEMPLATES]
    if unknown:
        raise ValueError(f"unknown QE template(s): {unknown}; known: {sorted(QE_TEMPLATES)}")
    primary = templates[0]

    # Gold label. `da` is the direct assessment used everywhere except the en-hi `mqm` config,
    # where 2,016 of 2,639 test rows (the WMT24 general-MT batch) carry an MQM score and NO DA.
    # Scoring those against `da_mean` would silently drop them. Both labels are written to every
    # record regardless, so a run scored one way can be rescored the other way without re-querying.
    if gold not in ("da", "mqm"):
        raise ValueError(f"gold must be 'da' or 'mqm', got {gold!r}")
    gold_field = "gold_da_mean" if gold == "da" else "gold_mqm_score"

    pair_obj = parse_pair(pair)
    dl = data_loader or DataLoader()
    cache = ResponseCache(Path(cache_dir) / f"qe__{provider.name}__{provider.model.replace('/', '__')}.jsonl")
    # max_tokens=8 suffices for models that emit a bare number, but REASONING models (e.g. sarvam-m
    # produce a chain-of-thought first and get cut off before the score. 512 lets them finish; models
    # that answer directly hit EOS early, so it costs them nothing. Override with IQE_QE_MAXTOK.
    cfg = GenerationConfig(max_tokens=int(os.environ.get("IQE_QE_MAXTOK", "512")), temperature=0.0)

    # Few-shot pool: drawn from the pair's TRAIN split (never the eval split) so the
    # challenge/test set is never leaked into the prompt. Source-disjoint by construction.
    #
    # FILTER BEFORE SAMPLING. The previous version sampled `shots` items and then dropped any
    # without a DA label, so a pair whose train split is DA-sparse silently got fewer examples
    # than asked for. Measured DA rates in the per-pair train splits: en-ta 0.469, en-hi 0.687,
    # en-mr 0.945, the rest 1.000. A nominal 4-shot en-ta prompt therefore carried about 1.9
    # examples on average, and the shot count varied by language, which is exactly the thing a
    # few-shot comparison must hold fixed. Filtering first makes `shots` mean what it says.
    fewshot_examples: list[Example] = []
    if shots > 0:
        pool = dl.load(pair, split="train", pair=pair)
        pool = pool[pool["da_mean"].notna()]
        if len(pool) < shots:
            raise ValueError(
                f"few-shot pool for {pair} has only {len(pool)} DA-bearing train rows, "
                f"fewer than the {shots} requested")
        pool = pool.sample(n=shots, random_state=seed + 10_000)
        for row in pool.itertuples(index=False):
            it_ = dl._row_to_item(row)
            fewshot_examples.append(
                Example(source=it_.source, mt=it_.mt, label=str(int(round(it_.da_mean)))))
        if len(fewshot_examples) != shots:
            raise ValueError(
                f"built {len(fewshot_examples)} few-shot examples for {pair}, expected {shots}")

    fs_sig = fewshot_signature(fewshot_examples)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumability: skip items already in the output.
    done_ids = _read_done_ids(out_path)
    is_resume = bool(done_ids)
    open_mode = "a" if is_resume else "w"

    with out_path.open(open_mode, encoding="utf-8") as fh:
        if not is_resume:
            fh.write(json.dumps({
                "_meta": {
                    "task": "qe", "subset": subset, "split": split, "pair": pair, "shots": shots,
                    "provider": provider.name, "model": provider.model,
                    "limit": limit, "seed": seed, "templates": list(templates),
                    "fewshot_signature": fs_sig,
                    "gold": gold, "gold_field": gold_field,
                    "generation": asdict(cfg),
                }
            }) + "\n")

        items = list(dl.iter_items(subset, split=split, pair=pair, limit=limit, seed=seed))
        for it in tqdm(items, desc=f"qe/{pair}/{provider.name}:{provider.model}"):
            if it.id in done_ids:
                continue
            preds: dict[str, float | None] = {}
            raws: dict[str, str] = {}
            for tkey in templates:
                sys_msg, usr_msg = render_qe(pair_obj, it.source, it.mt, fewshot=fewshot_examples, template=tkey)
                key = prompt_hash(
                    provider.name, provider.model, sys_msg, usr_msg,
                    task=f"qe:{tkey}", shots=shots, fewshot_signature=fs_sig,
                    **asdict(cfg),
                )
                response = cache.get(key)
                if response is None:
                    response = provider.complete(sys_msg, usr_msg, config=cfg)
                    cache.put(key, response)
                preds[tkey] = parse_qe_score(response)
                raws[tkey] = response
            fh.write(json.dumps({
                "id": it.id, "pair": it.pair, "task": "qe",
                "route": it.route, "is_curated": it.is_curated,
                # both golds always persisted, so a run can be rescored without re-querying
                "gold_da_mean": it.da_mean,
                "gold_mqm_score": it.mqm_score,
                "pred": preds, "raw": raws,
            }, ensure_ascii=False) + "\n")
            fh.flush()

    # Recompute summary from the full output file (including resumed prior items).
    golds: list[float | None] = []
    per_tpl_preds: dict[str, list[float | None]] = {t: [] for t in templates}
    n_total = 0
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in rec or "_summary" in rec:
                continue
            n_total += 1
            golds.append(rec.get(gold_field))
            pred_map = rec.get("pred") or {}
            for t in templates:
                per_tpl_preds[t].append(pred_map.get(t))

    by_template: dict[str, dict] = {}
    for t in templates:
        corr = correlation(per_tpl_preds[t], golds)
        n_parsed = sum(1 for p in per_tpl_preds[t] if p is not None)
        by_template[t] = {
            "pearson": corr.pearson, "spearman": corr.spearman, "kendall": corr.kendall,
            "n": corr.n, "n_parsed": n_parsed, "n_parse_failures": n_total - n_parsed,
        }

    n_gold = sum(1 for g in golds if g is not None)
    summary: dict = {
        # flat primary-template keys (back-compat with existing readers / tail checks)
        **by_template[primary],
        "n_attempted": n_total,
        "gold": gold,
        "gold_field": gold_field,
        "n_gold_present": n_gold,
        "n_gold_missing": n_total - n_gold,
        "primary_template": primary,
        "templates": list(templates),
        "by_template": by_template,
    }
    # prompt-sensitivity between the first two templates (the DA-vs-SQM story)
    if len(templates) >= 2:
        a, b = templates[0], templates[1]
        summary["prompt_sensitivity"] = {
            "compared": [a, b],
            **prompt_sensitivity(per_tpl_preds[a], per_tpl_preds[b]),
            "delta_pearson_vs_gold": by_template[a]["pearson"] - by_template[b]["pearson"],
            "delta_spearman_vs_gold": by_template[a]["spearman"] - by_template[b]["spearman"],
        }

    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_summary": summary}) + "\n")
    return summary
