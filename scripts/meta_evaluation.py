#!/usr/bin/env python3
"""Comprehensive QE meta-evaluation for the poster.

Segment (instance) level — per system, per pair, and macro-averaged across pairs:
    Pearson, Spearman, Kendall-tau-b   (vs gold DA)
System level — treating each language pair as a "system" (this QE data has ONE MT system per
pair, so the only available system axis is the language pair; true MT-system-level SPA is not
defined here and this is labelled accordingly):
    Pearson, Pairwise Accuracy (PA), Soft Pairwise Accuracy (SPA, Thompson et al. WMT24)
    over each judge/metric's per-pair mean score vs the gold per-pair mean DA.

Reads results/{qe_open,qe_closed}/*.jsonl (LLM judges; uses the gemba-da template) and
results/qe_metrics/*.jsonl (trained metrics). Writes results/analysis/qe_meta_*.csv and prints.

Usage: python scripts/meta_evaluation.py [results_dir] [shots]   (shots: 0 or 4, default 0)
"""
import csv, glob, json, os, sys, itertools, collections
import numpy as np
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
SHOTS = sys.argv[2] if len(sys.argv) > 2 else "0"
# NINE-pair default, matching the release and the paper's tab:qe. This previously defaulted to
# eight pairs (omitting enml) while the LLM judge files it reads are the CURRENT nine-pair
# challenge, so the defaults silently mixed two challenge-set generations in one macro average.
# tab:qe traces to qe_meta_s0_9pair_ext.csv, which could only be produced with env overrides;
# running this script as its own usage line documented regenerated a different, 8-pair file.
# Set IQE_PAIRS explicitly to reproduce the superseded 8-pair control.
PAIRS = os.environ.get(
    "IQE_PAIRS", "enhi,enmr,enta,engu,ente,enml,eten,neen,sien").split(",")
TMPL = "gemba-da"


def _load_judges(tmpl=TMPL):
    """system -> pair -> list[(pred, gold)] for the given template at the chosen shots."""
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f"{ROOT}/qe_open/*.jsonl") + glob.glob(f"{ROOT}/qe_closed/*.jsonl"):
        base = os.path.basename(f).replace(".jsonl", "")
        # name.s{shots}.{pair}
        try:
            name, srest = base.split(".s", 1)
            shot, pair = srest.split(".", 1)
        except ValueError:
            continue
        if shot != SHOTS or pair not in PAIRS:
            continue
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            g = r.get("gold_da_mean")
            p = (r.get("pred") or {}).get(tmpl) if isinstance(r.get("pred"), dict) else None
            if g is not None and p is not None:
                data[name][pair].append((float(p), float(g)))
    return data


#: trained-metric directory. `qe_metrics/` is the original 8-pair run on the pre-extension
#: challenge set; `qe_metrics_ext/` is the 9-pair run on the current extended challenge test, which
#: is the same population the LLM rows use. Set IQE_METRICS_DIR=qe_metrics_ext for the 9-pair table.
METRICS_DIR = os.environ.get("IQE_METRICS_DIR", "qe_metrics_ext")


def _load_metrics():
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f"{ROOT}/{METRICS_DIR}/*.jsonl"):
        base = os.path.basename(f).replace(".jsonl", "")
        try:
            name, pair = base.rsplit(".", 1)
        except ValueError:
            continue
        if pair not in PAIRS:
            continue
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            g, p = r.get("gold_da_mean"), r.get("pred_score")
            if g is not None and p is not None:
                data[name][pair].append((float(p), float(g)))
    return data


def _seg_corr(pairs_dict):
    """per-pair Pearson/Spearman/Kendall + macro average across pairs."""
    out = {}
    macro = {"pearson": [], "spearman": [], "kendall": []}
    for pair, pg in pairs_dict.items():
        if len(pg) < 10:
            continue
        p = np.array([x[0] for x in pg]); g = np.array([x[1] for x in pg])
        if p.std() == 0 or g.std() == 0:
            continue
        pe = stats.pearsonr(p, g).statistic
        sp = stats.spearmanr(p, g).statistic
        kd = stats.kendalltau(p, g).statistic
        out[pair] = (pe, sp, kd, len(pg))
        for k, v in zip(macro, (pe, sp, kd)):
            macro[k].append(v)
    # macro = mean across pairs; macro_sd = stdev across pairs (dispersion, NOT a CI on the corr —
    # a per-cell CI needs segment bootstrap). Reported as mean +/- sd in the summary.
    macro_sd = {k + "_sd": (float(np.std(v)) if v else float("nan")) for k, v in macro.items()}
    macro = {k: (float(np.mean(v)) if v else float("nan")) for k, v in macro.items()}
    macro.update(macro_sd)
    return out, macro


def _p_greater(a, b):
    """one-sided p that mean(a) > mean(b); Mann-Whitney U (rank-based, robust)."""
    if len(a) < 2 or len(b) < 2:
        return 0.5
    try:
        return float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)
    except ValueError:
        return 0.5


def _system_level(pairs_dict):
    """Pairs-as-systems: per-pair mean pred vs mean gold -> Pearson + PA + SPA."""
    syss = [p for p in PAIRS if len(pairs_dict.get(p, [])) >= 10]
    if len(syss) < 3:
        return None
    mean_pred = {s: np.mean([x[0] for x in pairs_dict[s]]) for s in syss}
    mean_gold = {s: np.mean([x[1] for x in pairs_dict[s]]) for s in syss}
    mp = np.array([mean_pred[s] for s in syss]); mg = np.array([mean_gold[s] for s in syss])
    pear = float(stats.pearsonr(mp, mg).statistic) if mp.std() and mg.std() else float("nan")
    pa_hits, spa_sum, n = 0, 0.0, 0
    for i, j in itertools.combinations(syss, 2):
        gi = np.array([x[1] for x in pairs_dict[i]]); gj = np.array([x[1] for x in pairs_dict[j]])
        pi = np.array([x[0] for x in pairs_dict[i]]); pj = np.array([x[0] for x in pairs_dict[j]])
        ph = _p_greater(gi, gj); pm = _p_greater(pi, pj)          # one-sided, same direction (i>j)
        spa_sum += 1.0 - abs(ph - pm)                              # SPA contribution
        pa_hits += (np.sign(mean_gold[i] - mean_gold[j]) == np.sign(mean_pred[i] - mean_pred[j]))
        n += 1
    return {"pearson": pear, "PA": pa_hits / n, "SPA": spa_sum / n, "n_systems": len(syss)}


def main():
    judges = _load_judges()
    metrics = _load_metrics()
    rows = []
    order = ["BharatGPT-3B-Indic", "Llama-3.2-3B-Instruct", "tiny-aya-fire", "sarvam-m",
             "sarvam-translate", "gpt-5.5", "cometkiwi-da", "cometkiwi-xl", "xcomet-xl"]
    allsys = {**judges, **metrics}
    for name in order + [s for s in allsys if s not in order]:
        if name not in allsys:
            continue
        seg, macro = _seg_corr(allsys[name])
        syslvl = _system_level(allsys[name])
        rows.append((name, seg, macro, syslvl))

    # print segment-level Kendall per pair
    print(f"=== SEGMENT-LEVEL Kendall-tau-b vs gold DA (shots={SHOTS}, {TMPL}) ===")
    print(f"{'system':<24}" + "".join(f"{p:>7}" for p in PAIRS) + f"{'macroK':>8}{'macroP':>8}{'macroS':>8}")
    for name, seg, macro, _ in rows:
        line = "".join(f"{seg[p][2]:>7.3f}" if p in seg else f"{'-':>7}" for p in PAIRS)
        print(f"{name:<24}{line}{macro['kendall']:>8.3f}{macro['pearson']:>8.3f}{macro['spearman']:>8.3f}")

    print(f"\n=== SYSTEM-LEVEL (pairs-as-systems; single MT/pair, see caveat) shots={SHOTS} ===")
    print(f"{'system':<24}{'Pearson':>9}{'PA':>7}{'SPA':>7}{'#sys':>6}")
    for name, _, _, sl in rows:
        if sl:
            print(f"{name:<24}{sl['pearson']:>9.3f}{sl['PA']:>7.3f}{sl['SPA']:>7.3f}{sl['n_systems']:>6}")

    # PROMPT ROBUSTNESS: macro Spearman (mean +/- sd across pairs) for BOTH prompts + the gap.
    # Metrics have no prompt -> shown as n/a. mean+/-sd is dispersion across pairs, not a CI.
    da = _load_judges("gemba-da"); sqm = _load_judges("gemba-sqm")
    print(f"\n=== PROMPT ROBUSTNESS — macro Spearman mean+/-sd across pairs (shots={SHOTS}) ===")
    print(f"{'system':<24}{'GEMBA-DA':>16}{'GEMBA-SQM':>16}{'|DA-SQM|':>10}")
    for name, _, _, _ in rows:
        if name not in da and name not in sqm:
            continue
        _, md = _seg_corr(da.get(name, {}))
        _, ms = _seg_corr(sqm.get(name, {}))
        d, s = md.get("spearman", float("nan")), ms.get("spearman", float("nan"))
        if name in metrics:   # trained metric: prompt-free
            print(f"{name:<24}{d:>10.3f}{'(no prompt)':>22}")
        else:
            gap = abs(d - s) if (d == d and s == s) else float("nan")
            print(f"{name:<24}{d:>8.3f}+/-{md['spearman_sd']:<5.3f}{s:>8.3f}+/-{ms['spearman_sd']:<5.3f}{gap:>10.3f}")

    os.makedirs(f"{ROOT}/analysis", exist_ok=True)
    with open(f"{ROOT}/analysis/qe_meta_s{SHOTS}{os.environ.get('IQE_META_SUFFIX','')}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "shots"] + [f"kendall_{p}" for p in PAIRS] +
                   ["macro_pearson", "macro_spearman", "macro_kendall",
                    "syslvl_pearson", "syslvl_PA", "syslvl_SPA", "n_systems"])
        for name, seg, macro, sl in rows:
            w.writerow([name, SHOTS] +
                       [f"{seg[p][2]:.4f}" if p in seg else "" for p in PAIRS] +
                       [f"{macro['pearson']:.4f}", f"{macro['spearman']:.4f}", f"{macro['kendall']:.4f}"] +
                       ([f"{sl['pearson']:.4f}", f"{sl['PA']:.4f}", f"{sl['SPA']:.4f}", sl['n_systems']] if sl else ["", "", "", ""]))
    print(f"\nwrote {ROOT}/analysis/qe_meta_s{SHOTS}.csv")


if __name__ == "__main__":
    main()
