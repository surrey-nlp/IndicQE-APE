#!/usr/bin/env python3
"""Build unified/en-ml.parquet from the finalised 10K En-Ml file (16 Jul 2026).

Supersedes the old 5K gated En-Ml: real 3-annotator DA + real human PE + real error
comments (-> tqr) + gold domain (news/finance/tech/legal/sports). The old RL layers
(wtag / mt_tokens / synthetic_description / error_categories) are merged onto the 5K
overlap by uid; the 5,000 new finance/tech/sports instances carry null word-level /
synthetic fields until those layers are generated. Schema is byte-identical to the
previous unified/en-ml.parquet; downstream curate_splits + make_hf_repo add
route/is_curated/human_ter and re-carve the challenge test.
"""
import json, sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_unified as bu
import normalize_domain as nd

#: The finalised 10K En-Ml annotation workbook (Finalized_10K_En-Ml-data.xlsx). It is not shipped
#: with the repo: pass its path as the first argument, or set ENML_10K_XLSX.
NEW = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENML_10K_XLSX", "")
OLD = os.path.join(os.path.dirname(__file__), "release_build/unified/en-ml.parquet")
OUT = OLD  # overwrite in place

NESTED = ["da_scores", "da_z_scores", "wtag", "mt_tokens", "tqr", "error_categories", "mqm_spans"]


def _parse(v):
    if isinstance(v, str) and v and v[0] in "[{":
        try: return json.loads(v)
        except Exception: return v
    return v


def main():
    if not NEW or not os.path.isfile(NEW):
        sys.exit("usage: python dataset/build_enml_10k.py <Finalized_10K_En-Ml-data.xlsx>"
                 "  (or set ENML_10K_XLSX)" + (f"\nnot found: {NEW}" if NEW else ""))
    new = pd.read_excel(NEW)
    old = pd.read_parquet(OLD)
    # old RL layers keyed by uid (recomputed uid to be safe)
    old_layers = {}
    for _, r in old.iterrows():
        u = bu.uid_of(r["source"], r["mt"])
        old_layers[u] = {k: _parse(r.get(k)) for k in
                         ("wtag", "mt_tokens", "synthetic_description", "error_categories", "explanation_model")}

    rows, seen = [], set()
    for _, x in new.iterrows():
        src, mt = str(x["Original"]), str(x["Translation"])
        u = bu.uid_of(src, mt)
        if u in seen:            # collapse the 1 normalise-key duplicate
            continue
        seen.add(u)
        das = [float(x[c]) for c in ("DA-A1", "DA-A2", "DA-A3") if pd.notna(x.get(c))]
        tqr = [str(x[c]).strip() for c in ("Comment-A1", "Comment-A2", "Comment-A3")
               if bu._present(x.get(c))]
        lay = old_layers.get(u, {})
        r = bu._row("alope-rl", "enml", src, mt, x.get("index", ""), "train",
                    domain=nd._norm_one(x.get("Domain")),
                    da_scores=(das or None),
                    da_mean=(float(x["DA_mean"]) if pd.notna(x.get("DA_mean")) else np.nan),
                    n_annotators=(len(das) if das else np.nan),
                    post_edit=(str(x["Post-Edited Translation"]) if bu._present(x.get("Post-Edited Translation")) else None),
                    tqr=(tqr or None),
                    wtag=lay.get("wtag"), mt_tokens=lay.get("mt_tokens"),
                    synthetic_description=lay.get("synthetic_description"),
                    error_categories=lay.get("error_categories"),
                    explanation_model=lay.get("explanation_model"))
        rows.append(r)

    out = pd.DataFrame(rows)
    # per-instance flags (mirrors build_unified.merge finalisation)
    out["da_conflict"] = False
    out["sources"] = "alope-rl"
    out["has_da"] = out["da_mean"].notna()
    out["has_pe"] = out["post_edit"].map(bu._present)
    out["has_wtag"] = out["wtag"].map(bu._present)
    out["has_tqr"] = out["tqr"].map(bu._present)
    out["has_synth_expl"] = out["synthetic_description"].map(bu._present)
    out["has_domain"] = out["domain"].map(bu._present)
    out["has_mqm"] = out["mqm_spans"].map(bu._present)
    out["gated_rl"] = out["has_wtag"] | out["has_tqr"] | out["has_synth_expl"]
    if "_config" not in out: out["_config"] = None
    # contiguous cadence in stable uid order
    out = out.sort_values(["language_pair", "uid"]).reset_index(drop=True)
    seq = out.groupby("language_pair").cumcount()
    out["cadence_id"] = ("ALOPERL-ENML-" + seq.astype(str).str.zfill(6))
    # serialise nested cols to JSON strings (stable parquet typing) — match build_unified
    def _ser(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return json.dumps([xx.item() if hasattr(xx, "item") else xx for xx in v], ensure_ascii=False)
        return v
    for c in NESTED:
        if c in out: out[c] = out[c].map(_ser)
    # align column order to the previous schema exactly
    out = out.reindex(columns=list(old.columns))
    out.to_parquet(OUT)

    print("rows:", len(out), "(was", len(old), ")")
    print("cols match old:", list(out.columns) == list(old.columns))
    print("domains:", out["domain"].value_counts().to_dict())
    print("coverage: has_da %d  has_pe %d  has_domain %d  has_tqr %d  has_wtag %d  has_synth %d  gated_rl %d" % (
        out.has_da.sum(), out.has_pe.sum(), out.has_domain.sum(), out.has_tqr.sum(),
        out.has_wtag.sum(), out.has_synth_expl.sum(), out.gated_rl.sum()))
    print("n_annotators:", out.n_annotators.value_counts().to_dict())
    print("PE!=MT (real edits):", int((out.post_edit.astype(str).str.strip() != out.mt.astype(str).str.strip()).sum()))
    print("cadence sample:", out.cadence_id.iloc[0], "->", out.cadence_id.iloc[-1])


if __name__ == "__main__":
    main()
