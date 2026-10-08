from pathlib import Path

import numpy as np
import xarray as xr

import bris.plot_nowcast as plotting


def test_select_diverse_members_returns_four_members(tmp_path: Path) -> None:
    reference_time = np.datetime64("2026-10-07T21:05:00", "ns")
    lead_minutes = [5, 10, 20, 60, 90, 120]
    times = reference_time + np.asarray(lead_minutes, dtype="timedelta64[m]")
    member = np.arange(10, dtype=np.int32)
    base = np.arange(6 * 3 * 4, dtype=np.float32).reshape(1, 6, 3, 4)
    values = base * (member[:, None, None, None] + 1)
    dataset = xr.Dataset(
        data_vars={
            "lwe_precipitation_rate": (
                ("ensemble_member", "time", "y", "x"),
                values,
            ),
            "forecast_reference_time": reference_time,
        },
        coords={
            "ensemble_member": member,
            "time": times,
            "y": np.arange(3),
            "x": np.arange(4),
        },
    )
    source = tmp_path / "precipitation.nc"
    dataset.to_netcdf(source)

    selected = plotting.select_diverse_members(source, lead_minutes, spatial_stride=1)

    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert set(selected).issubset(set(range(10)))


def test_plot_nowcast_uses_one_static_and_one_animation_path(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path]] = []
    source = tmp_path / "precipitation.nc"
    static = tmp_path / "plot.png"
    animation = tmp_path / "plot.gif"

    monkeypatch.setattr(
        plotting,
        "select_diverse_members",
        lambda *_args, **_kwargs: [0, 2, 5, 9],
    )
    monkeypatch.setattr(
        plotting,
        "plot_member_evolution",
        lambda _source, output, *_args: calls.append(("static", output)),
    )
    monkeypatch.setattr(
        plotting,
        "animate_members_and_median",
        lambda _source, output, *_args: calls.append(("animation", output)),
    )

    result = plotting.plot_nowcast(source, static, animation_path=animation)

    assert result == (static, animation)
    assert calls == [("static", static), ("animation", animation)]
