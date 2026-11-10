"""APE task runner: prompt → generate → parse → BLEU/chrF/COMET-Kiwi/TER vs. PE.

Resumable: if `--out` already exists, items already post-edited (by `id`)
are skipped and the run continues. The summary line is recomputed at the end.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..cache import ResponseCache, fewshot_signature, prompt_hash
from ..data import DataLoader
from ..languages import parse_pair
from ..metrics import ape_evaluate
from ..models import LLMProvider
from ..models.base import GenerationConfig
from ..prompts import Example, parse_ape_response, render_ape, render_translate
from .qe import _read_done_ids


def run_ape(
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
    with_comet: bool = True,
    mode: str = "postedit",
    data_loader: DataLoader | None = None,
) -> dict:
    """Generate post-edits (mode='postedit') or from-scratch translations (mode='retranslate')
    and score them against the human references + reference-free trained metrics.

    Works on any config; rows without a human post-edit are skipped, so this naturally evaluates
    only the PE-bearing En->Indic pairs (en-hi/mr/ta) — keeping the dedicated-MT 'retranslate'
    track apples-to-apples with LLM post-editing on the same items. Few-shot exemplars (postedit
    only) come from the pair's TRAIN split (no leakage of the eval set)."""
    retranslate = mode == "retranslate"
    task_name = "retranslate" if retranslate else "ape"
    pair_obj = parse_pair(pair)
    dl = data_loader or DataLoader()
    cache = ResponseCache(Path(cache_dir) / f"{task_name}__{provider.name}__{provider.model.replace('/', '__')}.jsonl")
    cfg = GenerationConfig(max_tokens=512, temperature=0.0)

    fewshot_examples: list[Example] = []
    if shots > 0 and not retranslate:
        for ex in dl.iter_items(pair, split="train", pair=pair, limit=shots * 4, seed=seed + 10_000):
            if ex.pe is None:
                continue
            fewshot_examples.append(Example(source=ex.source, mt=ex.mt, label=ex.pe))
            if len(fewshot_examples) >= shots:
                break

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
                    "task": task_name, "mode": mode, "subset": subset, "split": split, "pair": pair,
                    "shots": shots, "provider": provider.name, "model": provider.model,
                    "limit": limit, "seed": seed,
                    "fewshot_signature": fs_sig,
                    "generation": asdict(cfg),
                }
            }) + "\n")

        items = list(dl.iter_items(subset, split=split, pair=pair, limit=limit, seed=seed))
        for it in tqdm(items, desc=f"ape/{pair}/{provider.name}:{provider.model}"):
            if it.id in done_ids:
                continue
            if it.pe is None:
                continue
            if retranslate:
                sys_msg, usr_msg = render_translate(pair_obj, it.source)
            else:
                sys_msg, usr_msg = render_ape(pair_obj, it.source, it.mt, fewshot=fewshot_examples)
            key = prompt_hash(
                provider.name, provider.model, sys_msg, usr_msg,
                task=task_name, shots=0 if retranslate else shots, fewshot_signature=fs_sig,
                **asdict(cfg),
            )
            response = cache.get(key)
            if response is None:
                response = provider.complete(sys_msg, usr_msg, config=cfg)
                cache.put(key, response)
            hyp = parse_ape_response(response)
            fh.write(json.dumps({
                "id": it.id, "pair": it.pair, "task": task_name, "route": it.route,
                "source": it.source, "mt": it.mt, "ref_pe": it.pe,
                "hyp_pe": hyp, "raw_response": response,
            }, ensure_ascii=False) + "\n")
            fh.flush()

    # Recompute corpus-level metrics from the full output.
    sources: list[str] = []
    hyps: list[str] = []
    refs: list[str] = []
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in rec or "_summary" in rec:
                continue
            if rec.get("ref_pe") is None or rec.get("hyp_pe") is None:
                continue
            sources.append(rec["source"])
            hyps.append(rec["hyp_pe"])
            refs.append(rec["ref_pe"])

    if not hyps:
        summary = {"bleu": 0.0, "chrf": 0.0, "ter": 0.0, "comet_kiwi": None, "n": 0}
    else:
        result = ape_evaluate(sources, hyps, refs, with_comet=with_comet)
        summary = {
            "bleu": result.bleu, "chrf": result.chrf, "ter": result.ter,
            "comet_kiwi": result.comet_kiwi, "n": result.n,
        }
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_summary": summary}) + "\n")
    return summary
