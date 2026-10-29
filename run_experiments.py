#!/usr/bin/env python3
"""Backwards-compatible entry point; equivalent to ``layered-defense run``.

Example::

    python run_experiments.py --provider openrouter --model google/gemini-2.5-flash
"""
import sys

from layered_defense.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
