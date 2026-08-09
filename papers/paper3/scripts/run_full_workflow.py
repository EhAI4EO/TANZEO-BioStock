#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_full_workflow.py

Runs the complete FCH -> AGB -> AGC pipeline in the correct dependency
order: FCH must be trained and predicted first, since both AGB and AGC
consume the resulting FCH raster as an auxiliary predictor band.

Usage
-----
    python scripts/run_full_workflow.py --config config/config.yaml
    python scripts/run_full_workflow.py --config config/config.yaml --skip-agc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import agb_mapping, agc_mapping, fch_mapping  # noqa: E402
from src import model_utils as mu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FCH -> AGB -> AGC workflow.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--skip-fch", action="store_true", help="Skip FCH (assumes fch_output_map already exists).")
    parser.add_argument("--skip-agb", action="store_true")
    parser.add_argument("--skip-agc", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_full_workflow")

    config = mu.load_config(args.config)

    if not args.skip_fch:
        logger.info("### Stage 1/3: FCH ###")
        fch_mapping.train(config)
        fch_mapping.predict(config)
    else:
        logger.info("Skipping FCH stage (--skip-fch).")

    fch_output = Path(config["paths"]["fch_output_map"])
    if not fch_output.exists():
        raise FileNotFoundError(
            f"FCH output map not found at {fch_output}. AGB/AGC require this "
            f"file; run the FCH stage first or check paths.fch_output_map."
        )

    if not args.skip_agb:
        logger.info("### Stage 2/3: AGB ###")
        agb_mapping.train(config)
        agb_mapping.predict(config)
    else:
        logger.info("Skipping AGB stage (--skip-agb).")

    if not args.skip_agc:
        logger.info("### Stage 3/3: AGC ###")
        agc_mapping.train(config)
        agc_mapping.predict(config)
    else:
        logger.info("Skipping AGC stage (--skip-agc).")

    logger.info("Full workflow complete.")


if __name__ == "__main__":
    main()
