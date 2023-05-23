import os
from logging import Logger
from pathlib import Path
from typing import Text

import hydra
from omegaconf import DictConfig

import pyrovelocity.data
from pyrovelocity.config import print_config_tree
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def preprocess(conf: DictConfig, logger: Logger) -> None:
    """Preprocess data.

    Args:
        conf (DictConfig): omegaconf configuration
        logger (Logger): PREPROCESS logger

    Examples:
        preprocess(conf.data_external, logger)
    """
    for data_set in conf.process_data:
        print_config_tree(conf.data_sets[data_set], logger, ())
        data_set_conf = conf.data_sets[data_set]
        data_path = data_set_conf.rel_path
        processed_path = data_set_conf.derived.rel_path
        process_method = data_set_conf.derived.process_method
        process_args = data_set_conf.derived.process_args
        thresh_histogram_path = data_set_conf.derived.thresh_histogram_path

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  downloaded data: {conf.paths.data_external}\n"
            f"  processed data: {conf.paths.data_processed}\n"
        )
        Path(conf.paths.data_external).mkdir(parents=True, exist_ok=True)
        Path(conf.paths.data_processed).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n\nPreprocessing {data_set} data :\n\n"
            f"  from external: {data_path}\n"
            f"  to processed: {processed_path}\n"
            f"  using method: {process_method}\n"
        )

        if os.path.isfile(processed_path) and os.access(processed_path, os.R_OK):
            logger.info(f"{processed_path} exists")
        else:
            logger.info(f"generating {processed_path} ...")
            process_method_fn = getattr(pyrovelocity.data, process_method)
            adata = process_method_fn(
                data=data_path,
                processed_path=processed_path,
                thresh_histogram_path=thresh_histogram_path,
                **process_args,
            )
            print_attributes(adata)

            if os.path.isfile(processed_path) and os.access(processed_path, os.R_OK):
                logger.info(f"successfully generated {processed_path}")
            else:
                logger.warn(f"cannot find and read {processed_path}")


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Preprocess data.
    Args:
        conf {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PREPROCESS", log_level=conf.base.log_level)
    preprocess(conf, logger)


if __name__ == "__main__":
    main()
