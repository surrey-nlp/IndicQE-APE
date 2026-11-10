#!/usr/bin/env python3
"""Verify the assembled IndicQE-APE repo: every README config's data_files exist and load.
Run from inside the repo dir BEFORE pushing:  python3 load_test.py
Validates via the parquet builder against the exact paths declared in README.md configs.
After pushing, load_dataset("surrey-nlp/IndicQE-APE", "<config>") works normally.
"""
import os, sys, yaml
from datasets import load_dataset
front = open("README.md").read().split("---")[1]
cfgs = yaml.safe_load(front)["configs"]
ok = True
for c in cfgs:
    name = c["config_name"]; files = {d["split"]: d["path"] for d in c["data_files"]}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        ok = False; print(f"FAIL {name:12} missing: {missing}"); continue
    try:
        ds = load_dataset("parquet", data_files=files)
        print(f"OK  {name:12} " + str({k: ds[k].num_rows for k in ds}))
    except Exception as e:
        ok = False; print(f"FAIL {name:12} {e}")
sys.exit(0 if ok else 1)
