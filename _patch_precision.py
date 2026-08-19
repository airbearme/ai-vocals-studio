#!/usr/bin/env python3
"""Rebuild precision_voice_cloning_system.py methods in place."""
import re

path = "precision_voice_cloning_system.py"
lines = open(path).read().splitlines()


def find_def(name):
    for i, ln in enumerate(lines):
        if re.match(r"\s*def " + re.escape(name) + r"\b", ln):
            return i
    raise RuntimeError(f"def {name} not found")


a = find_def("_advanced_training")
v = find_def("_validate_model_quality")
f = find_def("_finalize_clone")
g = find_def("generate_precision_vocals")
s = find_def("get_system_status")

adv = open("_padv.txt").read().rstrip("\n").splitlines()
val = open("_pval.txt").read().rstrip("\n").splitlines()
gen = open("_pgen.txt").read().rstrip("\n").splitlines()

new_lines = (lines[:a] + adv +
             lines[v:f] + val +
             lines[f:g] + gen +
             lines[s:])

open(path, "w").write("\n".join(new_lines) + "\n")
print(f"Patched: {len(lines)} -> {len(new_lines)} lines")
