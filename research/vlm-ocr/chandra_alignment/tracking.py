from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

SUPPORTED_TRACKERS = {"tensorboard", "wandb"}


def tracker_names(report_to: str) -> list[str]:
    names = [name.strip().lower() for name in report_to.split(",") if name.strip()]
    if names == ["none"] or not names:
        return []
    unknown = set(names) - SUPPORTED_TRACKERS
    if unknown:
        raise ValueError(
            f"Unsupported tracker(s): {', '.join(sorted(unknown))}; "
            "use tensorboard, wandb, both, or none"
        )
    return list(dict.fromkeys(names))


def project_configuration(output_dir: str | Path) -> ProjectConfiguration:
    output = Path(output_dir)
    return ProjectConfiguration(
        project_dir=str(output),
        logging_dir=str(output / "logs"),
    )


def init_trackers(
    accelerator: Accelerator,
    config: Any,
    output_dir: str | Path,
) -> None:
    names = tracker_names(config.report_to)
    if not names:
        return
    init_kwargs: dict[str, dict[str, Any]] = {}
    if "wandb" in names:
        wandb_options: dict[str, Any] = {"dir": str(output_dir)}
        if config.tracker_run_name:
            wandb_options["name"] = config.tracker_run_name
        init_kwargs["wandb"] = wandb_options
    accelerator.init_trackers(
        project_name=config.tracker_project_name,
        config=asdict(config),
        init_kwargs=init_kwargs,
    )
