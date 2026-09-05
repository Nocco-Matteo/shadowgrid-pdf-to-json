#!/usr/bin/env python3
"""Entry-point alternativo: pipeline run end-to-end su un PDF."""
from pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run"] + __import__("sys").argv[1:]))
