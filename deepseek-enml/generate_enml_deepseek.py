#!/usr/bin/env python3
"""
generate_enml_deepseek.py  —  DeepSeek-V4-Pro generator for IndicQE-APE en-ml.

WHAT IT DOES (one script, two paths, reuses old data + generates only what's missing)
--------------------------------------------------------------------------------------
The surrey-nlp/IndicQE-APE en-ml split has 10,000 rows. After the ALOPE-RL recovery (18 Jul 2026)
**5,000** rows carry a `synthetic_description` (+ `error_categories`; these ALOPE rows also have a
Gemini variant in `synthetic_description_gemini`). The other **5,000** — the tech/sports/finance
domain-expansion rows that are NOT in ALOPE — have no description. Every row has a *human*
`post_edit` — that is the APE reference, so the model NEVER sees it.

  Path A  (has_synth_expl == False, ~5,000)  -> GENERATE description + APE in one call
          The tech/sports/finance rows with no description. The model analyses the MT vs the
          source, writes `synthetic_description` and `error_categories`, then produces the
          post-edit. These get a DeepSeek-v4-pro description ONLY — no Gemini variant.

  Path B  (has_synth_expl == True,  ~5,000)  -> APE ONLY, reusing the recovered ALOPE data
          The model is GIVEN the existing `synthetic_description`, `error_categories`
          and the human `tqr` remarks, and produces only the post-edit.

Output is one JSONL line per row (resumable, keyed by `uid`). `deepseek_post_edit` is the
model's APE hypothesis (hyp_pe) to MATRA-score against the human `post_edit` (ref_pe); the
generated `synthetic_description`/`error_categories` fill the ~5,000 domain-expansion gaps.

MODEL / API (DeepSeek, OpenAI-compatible — verified against api-docs.deepseek.com 2026-07)
-----------------------------------------------------------------------------------------
  base_url          https://api.deepseek.com
  model             deepseek-v4-pro   (1M context, max output 384K, concurrency 500)
  thinking          extra_body={"thinking": {"type": "enabled"}}   (high-effort reasoning)
  reasoning_effort  "high"   (thinking mode ignores temperature/top_p — do not set them)
  json output       response_format={"type": "json_object"}  (+ the word "json" and an
                    example are in the prompt, as the docs require)
  retry             429 / 500 / 503 -> exponential backoff; 400/401/402/422 -> abort row

USAGE
-----
  export DEEPSEEK_API_KEY=sk-...              # from platform.deepseek.com/api_keys
  python generate_enml_deepseek.py --smoke     # 6 rows (3 per path), prints everything
  python generate_enml_deepseek.py             # full 10,000 -> enml_deepseek.jsonl
  python generate_enml_deepseek.py --split test --workers 32
  python generate_enml_deepseek.py --limit 50 --only A     # 50 path-A rows only

  pip install openai pandas huggingface_hub pyarrow
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------ data loading
HF_REPO = "surrey-nlp/IndicQE-APE"
ENML_PARQUETS = ["data/pairs/en-ml/train.parquet",
                 "data/pairs/en-ml/validation.parquet",
                 "data/pairs/en-ml/test.parquet"]


def load_enml(input_path: str | None, split: str) -> "list[dict]":
    """Load en-ml rows as plain dicts. Reads the HF parquet files directly with pandas
    (the `datasets` loader trips on a null/string schema mismatch in this config), or a
    local .parquet / .jsonl passed via --input."""
    import pandas as pd
    if input_path:
        if input_path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
            df = pd.DataFrame(rows)
        else:
            df = pd.read_parquet(input_path)
        if "_split" not in df.columns:
            df["_split"] = "local"
    else:
        from huggingface_hub import hf_hub_download
        frames = []
        for f in ENML_PARQUETS:
            sp = f.split("/")[-1].replace(".parquet", "")
            if split not in ("all", sp):
                continue
            p = hf_hub_download(HF_REPO, f, repo_type="dataset")
            d = pd.read_parquet(p); d["_split"] = sp
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    if split not in ("all",) and "_split" in df.columns and input_path is None:
        df = df[df["_split"] == split]
    return df.to_dict("records")


# ------------------------------------------------------------------ helpers
def _clean(v):
    """None/NaN/empty-ish -> None; lists -> list; else str."""
    if v is None:
        return None
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        vv = [str(x) for x in v if x is not None and str(x).strip()]
        return vv or None
    s = str(v).strip()
    if s in ("", "None", "nan", "[]", "null"):
        return None
    return s


def has_desc(row: dict) -> bool:
    if row.get("has_synth_expl") in (True, "true", "True", 1):
        return bool(_clean(row.get("synthetic_description")))
    return bool(_clean(row.get("synthetic_description")))


def extract_json(text: str) -> dict | None:
    """Parse the model's content as JSON; if it wrapped it in prose/fences, pull the object."""
    if not text:
        return None
    text = text.strip()
    for candidate in (text, re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ prompts
SYSTEM_PROMPT = (
    "You are an expert English-to-Malayalam translator and MT post-editor with native Malayalam "
    "fluency and formal legal/technical register. You correct machine-translation errors "
    "(mistranslation, omission, addition, grammar, fluency, terminology, named entities, numbers) "
    "so the result is adequate to the English source and natural in Malayalam. You always reply "
    "with a single strict json object and nothing else."
)

# Path A: no existing description -> produce description + categories + post-edit.
PROMPT_A = """\
Analyse the Malayalam machine translation of the English source, then post-edit it.

English source:
{source}

Malayalam machine translation (to be post-edited):
{mt}

Domain: {domain}
{tqr_block}
Do all of the following:
1. Write `synthetic_description`: a precise English paragraph explaining the specific errors in the
   machine translation relative to the source (quote the offending Malayalam and give the correct
   rendering). If the translation is already adequate, say so briefly.
2. Write `error_categories`: a comma-separated list of MQM-style categories that apply
   (e.g. "Mistranslation, Fluency Error, Omission, Named Entity"), or "No Error".
3. Write `post_edit`: the corrected Malayalam translation — adequate to the source and fluent.
   Output Malayalam script only for this field; change nothing that is already correct.

Reply with a single json object with exactly these keys:
{{"synthetic_description": "...", "error_categories": "...", "post_edit": "..."}}
"""

# Path B: description already exists -> reuse it, produce post-edit only.
PROMPT_B = """\
Post-edit the Malayalam machine translation using the source and the error analysis already
provided. Reuse the analysis; do not rewrite it.

English source:
{source}

Malayalam machine translation (to be post-edited):
{mt}

Domain: {domain}

Existing error analysis (synthetic_description):
{synthetic_description}

Error categories: {error_categories}
{tqr_block}
Produce `post_edit`: the corrected Malayalam translation — adequate to the source and fluent,
addressing the errors named above. Output Malayalam script only; change nothing already correct.

Reply with a single json object with exactly this key:
{{"post_edit": "..."}}
"""


def build_messages(row: dict):
    source = _clean(row.get("source")) or ""
    mt = _clean(row.get("mt")) or ""
    domain = _clean(row.get("domain")) or "general"
    tqr = _clean(row.get("tqr"))
    tqr_block = ""
    if tqr:
        remarks = tqr if isinstance(tqr, list) else [tqr]
        tqr_block = ("Human quality remarks on the MT (consider these):\n"
                     + "\n".join(f"- {r}" for r in remarks) + "\n")
    if has_desc(row):
        path = "B_ape_only"
        user = PROMPT_B.format(source=source, mt=mt, domain=domain,
                               synthetic_description=_clean(row.get("synthetic_description")) or "",
                               error_categories=_clean(row.get("error_categories")) or "",
                               tqr_block=tqr_block)
    else:
        path = "A_gen_desc_and_ape"
        user = PROMPT_A.format(source=source, mt=mt, domain=domain, tqr_block=tqr_block)
    return path, [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}]


# ------------------------------------------------------------------ one row
RETRYABLE = {429, 500, 503}


def process_row(row, client, args):
    uid = row.get("uid")
    path, messages = build_messages(row)
    last_err = None
    for attempt in range(args.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                reasoning_effort=args.reasoning_effort,     # "high"
                max_tokens=args.max_tokens,                 # generous; model max is 384000
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "enabled"},
                            "user_id": "indicqe-enml-archchana"},
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            parsed = extract_json(content)
            if not parsed or "post_edit" not in parsed or not str(parsed.get("post_edit", "")).strip():
                # JSON mode can occasionally return empty/partial content -> retry a couple of times
                last_err = f"unparseable_or_empty (finish={resp.choices[0].finish_reason})"
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 20)); continue
            out = {
                "uid": uid, "_split": row.get("_split"), "language_pair": row.get("language_pair"),
                "domain": _clean(row.get("domain")), "path": path,
                "used_existing_synth_desc": path.startswith("B"),
                "synthetic_description": (parsed.get("synthetic_description")
                                          if path.startswith("A")
                                          else _clean(row.get("synthetic_description"))),
                "error_categories": (parsed.get("error_categories")
                                     if path.startswith("A")
                                     else _clean(row.get("error_categories"))),
                "deepseek_post_edit": (parsed.get("post_edit") if parsed else None),
                "explanation_model": args.model if path.startswith("A") else row.get("explanation_model"),
                "model": args.model, "reasoning_effort": args.reasoning_effort,
                "finish_reason": resp.choices[0].finish_reason,
                "usage": (resp.usage.model_dump() if getattr(resp, "usage", None) else None),
                "raw_content": content if args.save_raw else None,
                "reasoning_content": (getattr(msg, "reasoning_content", None) if args.save_reasoning else None),
                "error": None if parsed else last_err,
            }
            return out
        except Exception as e:
            code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if code in RETRYABLE and attempt < args.max_retries:
                time.sleep(min(2 ** attempt, 30)); continue
            break
    return {"uid": uid, "_split": row.get("_split"), "path": path,
            "deepseek_post_edit": None, "error": last_err or "failed"}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro en-ml synthetic-description + APE generator")
    ap.add_argument("--input", default=None, help="local .parquet/.jsonl (default: pull en-ml from HF)")
    ap.add_argument("--output", default="enml_deepseek.jsonl")
    ap.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    ap.add_argument("--only", default="both", choices=["both", "A", "B"], help="restrict to a path")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--reasoning-effort", default="high", choices=["high", "max"])
    ap.add_argument("--max-tokens", type=int, default=65536, help="cap on output tokens (model max 384000)")
    ap.add_argument("--workers", type=int, default=16, help="concurrency (account limit is 500)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N eligible rows")
    ap.add_argument("--smoke", action="store_true", help="3 rows per path, verbose, no resume")
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--save-reasoning", action="store_true", help="store the chain-of-thought")
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"Set {args.api_key_env} (get a key at https://platform.deepseek.com/api_keys)")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    rows = load_enml(args.input, args.split)
    for r in rows:
        r["_pathAB"] = "B" if has_desc(r) else "A"
    if args.only != "both":
        rows = [r for r in rows if r["_pathAB"] == args.only]

    if args.smoke:
        a = [r for r in rows if r["_pathAB"] == "A"][:3]
        b = [r for r in rows if r["_pathAB"] == "B"][:3]
        rows = a + b
        args.resume = False
    elif args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.output):
        for l in open(args.output, encoding="utf-8"):
            try:
                rec = json.loads(l)
                if rec.get("deepseek_post_edit") and not rec.get("error"):
                    done.add(rec.get("uid"))
            except Exception:
                pass
    todo = [r for r in rows if r.get("uid") not in done]
    nA = sum(1 for r in todo if r["_pathAB"] == "A")
    nB = sum(1 for r in todo if r["_pathAB"] == "B")
    print(f"en-ml rows: {len(rows)} | already done: {len(done)} | to do: {len(todo)} "
          f"(A gen-desc+APE: {nA}, B APE-only: {nB}) | model {args.model} effort {args.reasoning_effort}")
    if not todo:
        print("nothing to do."); return

    lock = threading.Lock()
    n_ok = n_err = 0
    t0 = time.time()
    with open(args.output, "a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_row, r, client, args): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out = fut.result()
            with lock:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n"); fh.flush()
            ok = out.get("deepseek_post_edit") and not out.get("error")
            n_ok += bool(ok); n_err += (not ok)
            if args.smoke:
                print(f"\n=== {out.get('path')} uid={out.get('uid')} ok={bool(ok)} ===")
                if out.get("synthetic_description"):
                    print("desc:", str(out["synthetic_description"])[:300])
                print("post_edit:", str(out.get("deepseek_post_edit"))[:300])
                if out.get("error"): print("error:", out["error"])
            elif i % 20 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(todo)}  ok={n_ok} err={n_err}  {rate:.1f} rows/s", flush=True)
    print(f"done. ok={n_ok} err={n_err} -> {args.output}")
    if n_err:
        print("re-run the same command to retry only the failed/missing rows (resume is on).")


if __name__ == "__main__":
    main()
