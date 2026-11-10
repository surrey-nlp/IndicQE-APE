#!/usr/bin/env python3
"""Test-first curation for the unified IndicQE-APE dataset.

Philosophy (PI decisions, 8 Jun 2026): the test set is the scarce, curated, diagnostic
artifact — carve the hardest, most informative instances FIRST; everything else is train.
We do NOT inherit the old WMT splits.

Difficulty is HUMAN-GROUNDED (four backbone axes):
  A1 disagreement       std of per-annotator DA (genuine ambiguity), mid-band weighted
  A2 edit_effort        token TER of MT -> human post-edit (real fixing needed)
  A3 word_error         WTAG BAD-token density and/or hard error categories
                        (Mistranslation/Untranslated/Addition)
  A4 contradiction      cross-task disagreement on the SAME instance:
                          (DA high) & (BAD-density high)   — "looks scored well, riddled with errors"
                          (DA low)  & (TER low)            — "scored badly, barely edited"

Curation guarantees:
  * SOURCE-DISJOINT splits (a source sentence never spans train/test/dev).
  * POWER-FLOORED ADAPTIVE: each (pair x stratum) cell filled to >= --floor where data
    allows, capped at --cap; scarce cells (en-ml, X->En) take what exists, flagged low-power.
  * MATCHED EASY CONTROLS per pair so the test measures the easy->hard DROP.
  * DEV mirrors the test difficulty distribution (per pair, DA-band stratified).

Inputs: unified parquet dir (from build_unified.py). Outputs: splits/ + manifests + report.

Usage:
  python3 curate_splits.py --unified-dir <build>/unified --out-dir <build>/curated \
      [--floor 100] [--cap 300] [--control-ratio 0.5] [--dev-frac 0.3] [--seed 13]
"""
import argparse, glob, json, os, re, sys, unicodedata
import numpy as np
import pandas as pd

AXES = ["A1_disagreement", "A2_edit_effort", "A3_word_error", "A4_contradiction"]
HARD_ERR = {"mistranslation", "untranslated", "addition"}
_ws = re.compile(r"\s+")
# MUST byte-match build_unified.py:norm_match so source-disjointness uses the SAME source
# key the uid/sanity-checker use (else en-dash vs hyphen splits one source across train/test).
_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'",
              "–": "-", "—": "-"}


def norm_match(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("<EOS>", "").replace("<eos>", "")
    for a, b in _QUOTE_MAP.items():
        s = s.replace(a, b)
    return _ws.sub("", s)


def jloads(v):
    """Parse a released list column, whichever way it is encoded.

    Two encodings are in use. `da_scores` ships as a JSON list, and `wtag` as a space-separated
    string, "BAD OK OK OK BAD ...". The space-separated branch is last so a genuine JSON list
    still parses as JSON.
    """
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        j = json.loads(s)
        return list(j) if isinstance(j, (list, tuple)) else None
    except Exception:
        return s.split()


def ter(mt, pe):
    a, b = str(mt).split(), str(pe).split()
    if not b:
        return np.nan
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1] / len(b)


def bad_density(wtag):
    w = jloads(wtag)
    if not w:
        return np.nan
    toks = [str(t).upper() for t in w]
    bad = sum(t.startswith("BAD") for t in toks)
    return bad / max(1, len(toks))


def has_hard_error(cats):
    if cats is None or (isinstance(cats, float) and cats != cats):
        return False
    s = str(cats).lower()
    return any(h in s for h in HARD_ERR)


# --------------------------------------------------------------------------- #
def compute_signals(df):
    df = df.copy()
    def _std(s):
        s = jloads(s)
        if not s or len(s) < 2:
            return np.nan
        try:
            vals = [float(x) for x in s if str(x).strip() not in ("", "nan", "None")]
        except (ValueError, TypeError):
            return np.nan
        return float(np.std(vals)) if len(vals) > 1 else np.nan
    df["da_std"] = df["da_scores"].map(_std)
    df["human_ter"] = df.apply(
        lambda r: ter(r["mt"], r["post_edit"]) if r.get("has_pe") else np.nan, axis=1)
    df["bad_density"] = df["wtag"].map(bad_density)
    df["hard_error"] = df["error_categories"].map(has_hard_error)
    df["src_key"] = df["source"].map(norm_match)
    return df


def _zwithin(df, col):
    return df.groupby("language_pair")[col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9) if s.notna().any() else s)


def axis_flags(df, q=0.75):
    df = df.copy()
    def topq(col):
        return df.groupby("language_pair")[col].transform(
            lambda s: s >= s.quantile(q) if s.notna().any() else False).fillna(False)
    midband = df["da_mean"].between(35, 75)
    df["A1_disagreement"] = topq("da_std") & midband
    df["A2_edit_effort"] = topq("human_ter")
    df["A3_word_error"] = topq("bad_density") | df["hard_error"]
    zda = _zwithin(df, "da_mean")
    zbad = _zwithin(df, "bad_density")
    zter = _zwithin(df, "human_ter")
    contra_a = (zda > 0.5) & (zbad > 0.5)                 # scored well yet error-dense
    contra_b = (df["da_mean"] <= df.groupby("language_pair")["da_mean"].transform(
        lambda s: s.quantile(0.3))) & (zter < -0.5)        # scored badly yet barely edited
    df["A4_contradiction"] = contra_a.fillna(False) | contra_b.fillna(False)
    # composite difficulty score for ranking (mean of available z-signals)
    df["difficulty"] = pd.concat([
        _zwithin(df, "da_std").fillna(0), zter.fillna(0), zbad.fillna(0),
        df["A4_contradiction"].astype(float)], axis=1).mean(axis=1)
    df["n_axes"] = df[AXES].sum(axis=1)
    return df


# En->Indic pairs we curate. X->En (et-en, ne-en, si-en) are NOT curated: they come from
# the Sheffield MLQE-PE repo and are passed through with their original splits (PI, 8 Jun).
CURATED_PAIRS = ["en-hi", "en-mr", "en-ta", "en-te", "en-gu", "en-ml"]
BANDS = [0, 35, 55, 75, 101]


def carve(df, floor, cap, control_ratio, dev_frac, seed, target_test=0,
          min_pair_test=1000, curated_pairs=None):
    curated = set(curated_pairs or CURATED_PAIRS)
    rng = np.random.RandomState(seed)
    df = df.reset_index(drop=True)
    df["orig_split"] = df["split"] if "split" in df.columns else "train"
    df["split"] = "train"
    df["route"] = ""
    assigned_src = {}      # src_key -> split (enforces source-disjointness)

    def claim(srcs, split):
        for s in srcs:
            assigned_src[s] = split

    def topup_pair(pair, n_need, route):
        """Add ~n_need test instances for one pair from its train pool, matched to that
        pair's current test DA-band distribution; source-disjoint. Returns added count."""
        if n_need <= 0:
            return 0
        tnow = df[(df.language_pair == pair) & (df.split == "test")]
        dist = (pd.cut(tnow["da_mean"], BANDS, labels=False).value_counts(normalize=True)
                if len(tnow) else pd.Series({1: .5, 2: .5}))
        pool = df[(df.language_pair == pair) & (df.split == "train")
                  & (~df["src_key"].isin(assigned_src))].copy()
        pool["band"] = pd.cut(pool["da_mean"], BANDS, labels=False)
        picks, added = [], 0
        for b, frac in dist.items():
            want = int(round(n_need * frac))
            cand_sk = list(pool[pool.band == b]["src_key"].unique())
            rng.shuffle(cand_sk)
            n = 0
            for sk in cand_sk:
                if n >= want:
                    break
                picks.append(sk); n += int((pool["src_key"] == sk).sum())
            added += n
        # if banding under-fills (sparse pair), top up from anything left
        if added < n_need:
            leftover = [sk for sk in pool["src_key"].unique() if sk not in picks]
            rng.shuffle(leftover)
            for sk in leftover:
                if added >= n_need:
                    break
                picks.append(sk); added += int((pool["src_key"] == sk).sum())
        claim(picks, "test")
        sel = df["src_key"].isin(picks) & (df.split == "train")
        df.loc[sel, "route"] = route
        df.loc[df["src_key"].isin(picks), "split"] = "test"
        return added

    # ---- 0) X->En passthrough: keep MLQE-PE split but assign each SOURCE-GROUP to one
    #         split (most-held-out among its rows) so the passthrough stays source-disjoint.
    unc = ~df["language_pair"].isin(curated)
    df.loc[unc, "route"] = "uncurated"
    rank = {"test": 0, "dev": 1, "validation": 1, "train": 2, "unknown": 3}
    u = df[unc].copy()
    u["r"] = u["orig_split"].map(rank).fillna(3)
    grp_split = (u.sort_values("r").groupby("src_key")["orig_split"].first()
                 .replace({"validation": "dev"}))
    df.loc[unc, "split"] = df.loc[unc, "src_key"].map(grp_split).fillna("train")
    claim(df.loc[unc, "src_key"].tolist(), "uncurated")

    # ---- 1) challenge TEST: per (curated pair, axis) hardest source-groups to floor/cap
    for pair in curated:
        gp = df[df.language_pair == pair]
        for axis in AXES:
            cand = gp[gp[axis] & ~gp["src_key"].isin(assigned_src)]
            if cand.empty:
                continue
            order = (cand.groupby("src_key")["difficulty"].max()
                     .sort_values(ascending=False).index)
            taken, n = [], 0
            for sk in order:
                if n >= cap:
                    break
                taken.append(sk); n += int((cand["src_key"] == sk).sum())
            claim(taken, "test")
            idx = df["src_key"].isin(taken) & df[axis]
            df.loc[idx & (df["route"] == ""), "route"] = axis
            df.loc[df["src_key"].isin(taken), "split"] = "test"

    # ---- 2) matched EASY CONTROLS per curated pair ----
    for pair in curated:
        gp = df[df.language_pair == pair]
        n_ctrl = int(((df.language_pair == pair) & (df.split == "test")).sum() * control_ratio)
        easy = gp[(~gp["src_key"].isin(assigned_src)) & (gp["n_axes"] == 0)].sort_values("difficulty")
        order = easy.groupby("src_key")["difficulty"].min().sort_values().index
        taken, n = [], 0
        for sk in order:
            if n >= n_ctrl:
                break
            taken.append(sk); n += int((easy["src_key"] == sk).sum())
        claim(taken, "test")
        df.loc[df["src_key"].isin(taken), ["split", "route"]] = ["test", "easy_control"]

    # ---- 3) PER-PAIR FLOOR: every curated pair's test >= min_pair_test ----
    for pair in curated:
        cur = int(((df.language_pair == pair) & (df.split == "test")).sum())
        if cur < min_pair_test:
            topup_pair(pair, min_pair_test - cur, "pair_floor")

    # ---- 4) GLOBAL TARGET: grow curated test to target_test, spread over curated pairs ----
    if target_test:
        cur_total = int(((df.split == "test") & (df.language_pair.isin(curated))).sum())
        need = target_test - cur_total
        if need > 0:
            # weight remaining need by each pair's leftover train availability
            avail = {p: int(((df.language_pair == p) & (df.split == "train")
                             & (~df["src_key"].isin(assigned_src))).sum()) for p in curated}
            tot = sum(avail.values()) or 1
            for p in curated:
                topup_pair(p, int(round(need * avail[p] / tot)), "distributional")

    # ---- 5) DEV mirroring test DA-band distribution, per curated pair ----
    for pair in curated:
        test_p = df[(df.language_pair == pair) & (df.split == "test")]
        n_dev = int(len(test_p) * dev_frac)
        if n_dev < 10:
            continue
        target = pd.cut(test_p["da_mean"], BANDS, labels=False).value_counts(normalize=True)
        pool = df[(df.language_pair == pair) & (df.split == "train")
                  & (~df["src_key"].isin(assigned_src))]
        pool_bands = pd.cut(pool["da_mean"], BANDS, labels=False)
        picks = []
        for b, frac in target.items():
            cand_sk = list(pool[pool_bands == b]["src_key"].unique())
            rng.shuffle(cand_sk)
            n, taken = 0, []
            for sk in cand_sk:
                if n >= int(n_dev * frac):
                    break
                taken.append(sk); n += int((pool["src_key"] == sk).sum())
            picks += taken
        claim(picks, "dev")
        df.loc[df["src_key"].isin(picks), "split"] = "dev"

    df.loc[(df.split == "test") & (df.route == ""), "route"] = "co_selected"
    return df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--target-test", type=int, default=10000,
                    help="grow curated test to this total via distributional top-up (0 = off)")
    ap.add_argument("--min-pair-test", type=int, default=1000,
                    help="minimum test size per curated En->Indic pair")
    ap.add_argument("--control-ratio", type=float, default=0.5)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.unified_dir, "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f">> {len(df)} unified instances; computing difficulty ...", file=sys.stderr)
    df = axis_flags(compute_signals(df), q=0.75)
    print(">> carving test-first ...", file=sys.stderr)
    df = carve(df, args.floor, args.cap, args.control_ratio, args.dev_frac, args.seed,
               target_test=args.target_test, min_pair_test=args.min_pair_test)

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["test", "dev", "train"]:
        sub = df[df.split == split]
        sub.to_parquet(os.path.join(args.out_dir, f"{split}.parquet"))
    # blind test (inputs only)
    blind_cols = ["uid", "cadence_id", "language_pair", "source", "mt", "domain", "route"]
    df[df.split == "test"][blind_cols].to_parquet(os.path.join(args.out_dir, "test_blind.parquet"))

    # ---- diagnostics ----
    test = df[df.split == "test"]
    print("\n===== SPLIT SIZES (source-disjoint) =====")
    print(df.groupby(["language_pair", "split"]).size().unstack(fill_value=0).to_string())
    # source-disjointness check
    overlap = 0
    for s in ["test", "dev", "train"]:
        for t in ["test", "dev", "train"]:
            if s < t:
                a = set(df[df.split == s]["src_key"]); b = set(df[df.split == t]["src_key"])
                overlap += len(a & b)
    print(f"\nsource-disjointness violations: {overlap} (must be 0)")

    print("\n===== CHALLENGE-TEST COMPOSITION (route x pair) =====")
    comp = test.groupby(["route", "language_pair"]).size().unstack(fill_value=0)
    print(comp.to_string())
    # power flags — only real diagnostic strata (the four axes)
    cells = test[test.route.isin(AXES)].groupby(["language_pair", "route"]).size()
    low = cells[cells < args.floor]
    print(f"\nlow-power diagnostic cells (<{args.floor}): {len(low)}")
    if len(low):
        print(low.to_string())

    print("\n===== TEST vs TRAIN difficulty (validation) =====")
    df["_da"] = pd.to_numeric(df["da_mean"], errors="coerce")
    val = df.groupby("split").agg(
        n=("uid", "size"), mean_DA=("_da", "mean"), DA_std=("_da", "std"),
        disagree=("da_std", "mean"), mean_TER=("human_ter", "mean"),
        bad_dens=("bad_density", "mean")).round(3)
    print(val.to_string())
    df[["uid", "cadence_id", "language_pair", "split", "route", "difficulty", "n_axes"]
       ].to_csv(os.path.join(args.out_dir, "split_manifest.csv"), index=False)
    print(f"\nwritten to {args.out_dir}/ (test/dev/train.parquet, test_blind, split_manifest.csv)")


if __name__ == "__main__":
    main()
