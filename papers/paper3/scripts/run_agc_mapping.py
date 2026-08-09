#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_agc_mapping.py

Command-line entry point for the Above-Ground Carbon (AGC) mapping stage.

**Depends on FCH**: run `run_fch_mapping.py` first -- this stage requires
`paths.fch_output_map` to already exist, since FCH is used as an
auxiliary predictor band.

Usage
-----
    python scripts/run_agc_mapping.py --config config/config.yaml
    python scripts/run_agc_mapping.py --config config/config.yaml --stage train
    python scripts/run_agc_mapping.py --config config/config.yaml --stage predict
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import agc_mapping  # noqa: E402
from src import model_utils as mu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AGC mapping workflow.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--stage", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_agc_mapping")

    config = mu.load_config(args.config)

    fch_output = Path(config["paths"]["fch_output_map"])
    if not fch_output.exists():
        logger.warning(
            "FCH output map not found at %s. AGC training/prediction requires "
            "the FCH stage to be run first (see README.md).", fch_output,
        )

    if args.stage in ("train", "all"):
        logger.info("=== AGC: training stage ===")
        result = agc_mapping.train(config)
        logger.info(
            "AGC training complete. Selected k=%d features. Test metrics: %s",
            result["best_k"], result["metrics"],
        )

    if args.stage in ("predict", "all"):
        logger.info("=== AGC: wall-to-wall prediction stage ===")
        output_path = agc_mapping.predict(config)
        logger.info("AGC map written to %s", output_path)


if __name__ == "__main__":
    main()
