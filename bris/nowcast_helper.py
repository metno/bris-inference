from __future__ import annotations

import argparse
import re
import shlex
import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from string import Template

import pandas as pd


def render_template(template_file: Path, output_file: Path, values: dict[str, str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(Template(template_file.read_text()).substitute(values))


def shell_assign(name: str, value: str | Path) -> str:
    return f"{name}={shlex.quote(str(value))}"


def parse_config_datetime(value: str | None) -> datetime:
    if value is None or str(value).strip() == "":
        return datetime.now(UTC)
    text = str(value).strip().removesuffix("Z")
    for fmt in (
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H:%M:%S",
        "%Y%m%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y%m%d",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    raise SystemExit(f"Could not parse run_datetime: {value!r}")


def _looks_like_timestamp(value: str) -> bool:
    return bool(
        re.match(r"^[0-9]{8}T[0-9]{6}Z?$", value)
        or re.match(r"^[0-9]{8}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z?$", value)
        or re.match(
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z?)?$",
            value,
        )
        or re.match(r"^[0-9]{8}$", value)
    )


def expand_path_template(
    value: str | Path,
    run_dt: datetime,
    base_dir: Path | None = None,
) -> Path:
    mapping = {
        "YYYYmmddTHHMMSS": run_dt.strftime("%Y%m%dT%H%M%S"),
        "YYYYMMDDTHHMMSS": run_dt.strftime("%Y%m%dT%H%M%S"),
        "YYYYmmddHHMMSS": run_dt.strftime("%Y%m%d%H%M%S"),
        "YYYYMMDDHHMMSS": run_dt.strftime("%Y%m%d%H%M%S"),
        "YYYYmmddHHMM": run_dt.strftime("%Y%m%d%H%M"),
        "YYYYMMDDHHMM": run_dt.strftime("%Y%m%d%H%M"),
        "YYYYmmdd": run_dt.strftime("%Y%m%d"),
        "YYYYMMDD": run_dt.strftime("%Y%m%d"),
        "YYYY": run_dt.strftime("%Y"),
        "mm": run_dt.strftime("%m"),
        "dd": run_dt.strftime("%d"),
        "HH": run_dt.strftime("%H"),
        "MMIN": run_dt.strftime("%M"),
    }
    text = str(value)
    for key, replacement in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace("${" + key + "}", replacement)
        text = text.replace("{" + key + "}", replacement)
    for key in ("YYYYmmddTHHMMSS", "YYYYMMDDTHHMMSS", "YYYYmmdd", "YYYYMMDD"):
        text = text.replace(key, mapping[key])
    path = Path(text).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _path_config_keys() -> set[str]:
    return {
        "checkpoint",
        "frost_credentials",
        "lightning_dataset",
        "lightning_grid_definition",
        "lightning_template",
        "meps_dataset",
        "meps_root",
        "netatmo_dataset",
        "netatmo_root",
        "radar_dataset",
        "radar_root",
    }


def load_nowcast_config(args: argparse.Namespace) -> None:
    import yaml

    config_file = args.config.expanduser().resolve(strict=False)
    if not config_file.is_file():
        raise SystemExit(f"Config file not found: {config_file}")
    config = yaml.safe_load(config_file.read_text()) or {}
    required = ("input_files_path", "output_file_path", "number_ens")
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"Missing required config key(s): {', '.join(missing)}")

    run_value = config.get("run_datetime")
    radar_arg = str(config.get("radar_arg", "") or "").strip()
    if run_value is None and radar_arg and _looks_like_timestamp(radar_arg):
        run_value = radar_arg
    run_dt = parse_config_datetime(run_value)
    config_dir = config_file.parent
    number_ens = int(config["number_ens"])
    if number_ens <= 0:
        raise SystemExit("number_ens must be positive")

    if radar_arg and ("/" in radar_arg or radar_arg.endswith(".nc")):
        radar_arg = str(expand_path_template(radar_arg, run_dt, config_dir))
    elif re.match(r"^[0-9]{8}$", radar_arg) and run_value is not None:
        radar_arg = run_dt.strftime("%Y%m%dT%H:%M")

    values: dict[str, str | int | Path] = {
        "NOWCAST_CONFIG_FILE": config_file,
        "NOWCAST_RADAR_ARG": radar_arg,
        "BRIS_INPUT_DIR": expand_path_template(config["input_files_path"], run_dt, config_dir),
        "BRIS_OUTPUT_ROOT": expand_path_template(config["output_file_path"], run_dt, config_dir),
        "BRIS_NUM_MEMBERS": number_ens,
    }
    optional_env = {
        "checkpoint": "BRIS_CHECKPOINT",
        "checkpoint_sha256": "BRIS_CHECKPOINT_SHA256",
        "cycle_interval_minutes": "BRIS_CYCLE_INTERVAL_MINUTES",
        "dataloader_num_workers": "BRIS_DATALOADER_NUM_WORKERS",
        "dataloader_persistent_workers": "BRIS_DATALOADER_PERSISTENT_WORKERS",
        "dataloader_pin_memory": "BRIS_DATALOADER_PIN_MEMORY",
        "dataloader_prefetch_factor": "BRIS_DATALOADER_PREFETCH_FACTOR",
        "frost_credentials": "BRIS_FROST_CREDENTIALS",
        "inference_num_chunks": "BRIS_INFERENCE_NUM_CHUNKS",
        "lightning_dataset": "BRIS_LIGHTNING_DATASET",
        "lightning_grid_definition": "BRIS_LIGHTNING_GRID_DEFINITION",
        "lightning_template": "BRIS_LIGHTNING_TEMPLATE",
        "lightning_template_crs": "BRIS_LIGHTNING_TEMPLATE_CRS",
        "meps_backend": "BRIS_MEPS_BACKEND",
        "meps_dataset": "BRIS_MEPS_DATASET",
        "meps_root": "BRIS_MEPS_ROOT",
        "netatmo_backend": "BRIS_NETATMO_BACKEND",
        "netatmo_dataset": "BRIS_NETATMO_DATASET",
        "netatmo_root": "BRIS_NETATMO_ROOT",
        "output_cell_chunk": "BRIS_OUTPUT_CELL_CHUNK",
        "radar_dataset": "BRIS_RADAR_DATASET",
        "radar_endpoint": "BRIS_RADAR_ENDPOINT",
        "radar_root": "BRIS_RADAR_ROOT",
    }
    path_keys = _path_config_keys()
    for key, env_name in optional_env.items():
        value = config.get(key)
        if value in (None, ""):
            continue
        if key in path_keys:
            value = expand_path_template(value, run_dt, config_dir)
        elif isinstance(value, bool):
            value = int(value)
        values[env_name] = value

    for name, value in values.items():
        print(shell_assign(name, value))


def _radar_timestamp(radar_file: Path) -> datetime:
    match = re.search(r"([0-9]{8}T[0-9]{6})Z", radar_file.name)
    if not match:
        raise ValueError(f"Could not parse radar timestamp from {radar_file.name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=UTC
    )


def resolve_cycle(args: argparse.Namespace) -> None:
    interval = int(args.cycle_interval_minutes)
    if interval < 5 or 60 % interval:
        raise SystemExit("Cycle interval must be at least 5 minutes and divide 60 exactly")
    value = (args.run_time or "").strip()
    if value and Path(value).is_file():
        run_time = _radar_timestamp(Path(value))
    elif value:
        run_time = parse_config_datetime(value)
    else:
        run_time = None
        for radar_file in sorted(args.radar_root.rglob("*.nc"), reverse=True):
            try:
                candidate = _radar_timestamp(radar_file)
            except ValueError:
                continue
            if candidate.minute % interval == 0 and candidate.second == 0:
                run_time = candidate
                break
        if run_time is None:
            raise SystemExit(
                f"No radar cycle aligned to {interval} minutes found under {args.radar_root}"
            )
    if run_time.minute % interval or run_time.second:
        raise SystemExit(
            f"Nowcast reference time must be on a {interval}-minute boundary: {run_time}"
        )
    print(shell_assign("RUN_ISO", run_time.strftime("%Y-%m-%dT%H:%M:%S")))
    print(shell_assign("RUN_KEY", run_time.strftime("%Y%m%dT%H%M")))


def render_bris_config(args: argparse.Namespace) -> None:
    meps_end = pd.Timestamp(args.meps_iso)
    meps_start = meps_end - pd.Timedelta(hours=1)
    render_template(
        Path(str(files("bris").joinpath("schema/bris-nowcast.yaml"))),
        args.output,
        {
            "run_iso": args.run_iso,
            "checkpoint": str(args.checkpoint),
            "workdir": str(args.workdir),
            "meps_zarr": str(args.meps_zarr),
            "meps_start_iso": meps_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "meps_end_iso": meps_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "radar_dataset": str(args.radar_dataset),
            "netatmo_dataset": str(args.netatmo_dataset),
            "lightning_dataset": str(args.lightning_dataset),
            "dataloader_num_workers": str(args.dataloader_num_workers),
            "dataloader_prefetch_factor": str(args.dataloader_prefetch_factor),
            "dataloader_pin_memory": str(args.dataloader_pin_memory),
            "dataloader_persistent_workers": str(args.dataloader_persistent_workers),
            "num_members": str(args.num_members),
            "inference_num_chunks": str(args.inference_num_chunks),
        },
    )


def validate_bris_config(args: argparse.Namespace) -> None:
    from bris.utils import validate

    validate(str(args.config), raise_on_error=True)
    print(f"Validated BRIS config: {args.config}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BRIS operational nowcast helpers")
    subparsers = parser.add_subparsers(required=True)

    command = subparsers.add_parser("load-nowcast-config")
    command.add_argument("config", type=Path)
    command.set_defaults(func=load_nowcast_config)

    command = subparsers.add_parser("resolve-cycle")
    command.add_argument("run_time", nargs="?")
    command.add_argument("--radar-root", type=Path, required=True)
    command.add_argument("--cycle-interval-minutes", type=int, default=5)
    command.set_defaults(func=resolve_cycle)

    command = subparsers.add_parser("render-bris-config")
    command.add_argument("output", type=Path)
    command.add_argument("run_iso")
    command.add_argument("meps_iso")
    command.add_argument("radar_dataset", type=Path)
    command.add_argument("meps_zarr", type=Path)
    command.add_argument("netatmo_dataset", type=Path)
    command.add_argument("lightning_dataset", type=Path)
    command.add_argument("workdir", type=Path)
    command.add_argument("checkpoint", type=Path)
    command.add_argument("dataloader_num_workers")
    command.add_argument("dataloader_prefetch_factor")
    command.add_argument("dataloader_pin_memory")
    command.add_argument("dataloader_persistent_workers")
    command.add_argument("num_members")
    command.add_argument("inference_num_chunks")
    command.set_defaults(func=render_bris_config)

    command = subparsers.add_parser("validate-bris-config")
    command.add_argument("config", type=Path)
    command.set_defaults(func=validate_bris_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
