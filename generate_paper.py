#!/usr/bin/env python3
"""Backwards-compatible entry point; renders results.json into results_tables.tex.

``paper.tex`` includes that file, so the hand-written paper is never overwritten.
Equivalent to ``layered-defense report results.json --out results_tables.tex``.
"""
import sys

from layered_defense.cli import main

if __name__ == "__main__":
    args = sys.argv[1:] or ["results.json", "--out", "results_tables.tex"]
    sys.exit(main(["report", *args]))
