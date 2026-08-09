#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_fch_mapping.py

Command-line entry point for the Forest Canopy Height (FCH) mapping stage.

Usage
-----
    python scripts/run_fch_mapping.py --config config/config.yaml
    python scripts/run_fch_mapping.py --config config/config.yaml --stage train
    python scripts/run_fch_mapping.py --config config/config.yaml --stage predict
    python scripts/run_fch_mapping.py --config config/config.yaml --evaluate-baselines
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fch_mapping  # noqa: E402
from src import model_utils as mu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FCH mapping workflow.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--stage", choices=["train", "predict", "all"], default="all",
        help="Which stage(s) to run (default: all = train then predict).",
    )
    parser.add_argument(
        "--evaluate-baselines", action="store_true",
        help="Also run optional SVR/CatBoost/1D-CNN baseline comparisons (requires 'train' stage data).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_fch_mapping")

    config = mu.load_config(args.config)

    if args.stage in ("train", "all"):
        logger.info("=== FCH: training stage ===")
        result = fch_mapping.train(config)
        logger.info("FCH training complete. Test metrics: %s", result["metrics"])

        if args.evaluate_baselines:
            logger.info("=== FCH: optional baseline comparison ===")
            gdf_buffered = fch_mapping.prepare_training_geometries(config)
            stack, profile = fch_mapping.build_predictor_stack(config)
            from src import raster_utils as ru
            stack_path = Path(config["paths"]["fch_predictor_stack_20band"])
            X_train, y_train, X_test, y_test = fch_mapping.extract_training_samples(config, gdf_buffered, stack_path)
            from src import baseline_models as bm
            base_cfg = config.get("baselines", {})
            bm.run_all_baselines(
                X_train, y_train, X_test, y_test,
                include_catboost=base_cfg.get("include_catboost", True),
                include_cnn=base_cfg.get("include_cnn", True),
            )

    if args.stage in ("predict", "all"):
        logger.info("=== FCH: wall-to-wall prediction stage ===")
        output_path = fch_mapping.predict(config)
        logger.info("FCH map written to %s", output_path)


if __name__ == "__main__":
    main()
