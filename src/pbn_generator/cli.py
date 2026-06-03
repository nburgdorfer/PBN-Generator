from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from pbn_generator.config import PBNConfig
from pbn_generator.generator import PBNGenerator
from pbn_generator.logger import configure_logging


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    values: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    config = PBNConfig.from_mapping(values)
    PBNGenerator(config, base_dir=Path.cwd()).run()


if __name__ == "__main__":
    main()
