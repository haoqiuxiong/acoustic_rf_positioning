#!/usr/bin/env python3
"""RF backprojection matched filter from tutorial_rf_bp.ipynb with measurement and simulation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


C0 = 299_792_458.0
REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_UTILS_PATH = REPO_ROOT / "processing" / "tutorials" / "csi_plot_utils.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF backprojection matched filter with measurement and simulation."
    )
    parser.add_argument(
        "--experiment-id",
        default="EXP009",
        help="Experiment ID to localize. Default: EXP009.",
    )
    parser.add_argument(
        "--cycle-id",
        type=int,
        default=100,
        help="Cycle ID to localize. Default: 100.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Processed RF NetCDF path. Defaults to tutorial dataset discovery.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=920e6,
        help="Carrier frequency in Hz. Default: 920e6.",
    )
    parser.add_argument(
        "--path-loss-exponent",
        type=float,
        default=1.0,
        help="Path-loss exponent alpha. Default: 1.0.",
    )
    parser.add_argument(
        "--tx-height-m",
        type=float,
        default=None,
        help="Known TX height for 2D localization. Defaults to selected cycle rover_z.",
    )
    return parser.parse_args()


def load_csi_utils() -> Any:
    """Load csi_plot_utils module."""
    if not TUTORIAL_UTILS_PATH.exists():
        raise FileNotFoundError(f"Could not find tutorial utilities at {TUTORIAL_UTILS_PATH}")

    spec = importlib.util.spec_from_file_location("csi_plot_utils", TUTORIAL_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tutorial utilities from {TUTORIAL_UTILS_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["csi_plot_utils"] = module
    spec.loader.exec_module(module)
    return module


def build_search_grid(
    x_bounds_m: tuple[float, float] = (0.0, 8.0),
    y_bounds_m: tuple[float, float] = (0.0, 4.0),
    resolution_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D search grid."""
    x_min, x_max = float(x_bounds_m[0]), float(x_bounds_m[1])
    y_min, y_max = float(y_bounds_m[0]), float(y_bounds_m[1])
    xs = np.arange(x_min, x_max + resolution_m, resolution_m)
    ys = np.arange(y_min, y_max + resolution_m, resolution_m)
    return np.meshgrid(xs, ys, indexing="xy")


def channel_for_point(
    tx_x_m: float,
    tx_y_m: float,
    tx_z_m: float,
    antenna_xyz: np.ndarray,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Compute channel for a single TX point."""
    wavelength = C0 / float(frequency_hz)
    k = 2.0 * np.pi / wavelength

    dx = float(tx_x_m) - antenna_xyz[:, 0]
    dy = float(tx_y_m) - antenna_xyz[:, 1]
    dz = float(tx_z_m) - antenna_xyz[:, 2]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    distance = np.maximum(distance, 1e-6)
    amplitude = 1.0 / np.power(distance, float(path_loss_exponent))
    return amplitude * np.exp(-1j * k * distance)


def channel_image_map(
    antenna_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    tx_z_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Compute channel image map over grid."""
    wavelength = C0 / float(frequency_hz)
    k = 2.0 * np.pi / wavelength

    dx = grid_x[..., None] - antenna_xyz[:, 0][None, None, :]
    dy = grid_y[..., None] - antenna_xyz[:, 1][None, None, :]
    dz = float(tx_z_m) - antenna_xyz[:, 2][None, None, :]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    distance = np.maximum(distance, 1e-6)
    amplitude = 1.0 / np.power(distance, float(path_loss_exponent))
    return amplitude * np.exp(-1j * k * distance)


def matched_filter_image(
    y: np.ndarray,
    antenna_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    tx_z_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Compute matched-filter image: sum(conj(a(q)) * y)."""
    image_map = channel_image_map(
        antenna_xyz=antenna_xyz,
        grid_x=grid_x,
        grid_y=grid_y,
        tx_z_m=tx_z_m,
        frequency_hz=frequency_hz,
        path_loss_exponent=path_loss_exponent,
    )
    return np.sum(np.conjugate(image_map) * y[None, None, :], axis=2)


def estimate_xy_from_image(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    image: np.ndarray,
) -> tuple[float, float, complex, float]:
    """Estimate TX position from matched-filter image."""
    magnitude = np.abs(image)
    best_idx = int(np.nanargmax(magnitude))
    iy, ix = np.unravel_index(best_idx, magnitude.shape)
    return (
        float(grid_x[iy, ix]),
        float(grid_y[iy, ix]),
        complex(image[iy, ix]),
        float(magnitude[iy, ix]),
    )


def simulate_channel(
    tx_x_m: float,
    tx_y_m: float,
    tx_z_m: float,
    antenna_xyz: np.ndarray,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
    complex_gain: complex = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate channel with optional noise."""
    y = complex(complex_gain) * channel_for_point(
        tx_x_m=tx_x_m,
        tx_y_m=tx_y_m,
        tx_z_m=tx_z_m,
        antenna_xyz=antenna_xyz,
        frequency_hz=frequency_hz,
        path_loss_exponent=path_loss_exponent,
    )
    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = (
            noise_std
            * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
            / np.sqrt(2.0)
        )
        y = y + noise
    return y


def extract_measurement(
    ds: Any,
    csi: Any,
    experiment_id: str,
    cycle_id: int,
    antenna_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract antenna positions and CSI for a cycle."""
    snapshot = csi.extract_csi_snapshot(
        ds,
        experiment_id,
        int(cycle_id),
        antenna_positions=antenna_positions,
    )
    antenna_xyz = np.column_stack(
        [
            snapshot["antenna_x"].values.astype(float),
            snapshot["antenna_y"].values.astype(float),
            snapshot["antenna_z"].values.astype(float),
        ]
    )
    y = snapshot["csi_real"].values.astype(float) + 1j * snapshot["csi_imag"].values.astype(
        float
    )
    return antenna_xyz, y


def plot_image(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    image: np.ndarray,
    antenna_xyz: np.ndarray,
    tx_x_est: float,
    tx_y_est: float,
    title: str,
    true_xy_m: tuple[float, float] | None = None,
    xy_error_m: float | None = None,
) -> None:
    """Plot matched-filter image."""
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(
        np.abs(image),
        origin="lower",
        extent=[float(grid_x.min()), float(grid_x.max()), float(grid_y.min()), float(grid_y.max())],
        cmap="viridis",
        aspect="equal",
    )

    ax.scatter(
        antenna_xyz[:, 0],
        antenna_xyz[:, 1],
        marker="^",
        s=30,
        c="white",
        edgecolors="black",
        label="RX antennas",
    )

    if true_xy_m is not None:
        tx_x_true, tx_y_true = true_xy_m
        ax.scatter(
            [tx_x_true],
            [tx_y_true],
            marker="o",
            s=90,
            c="red",
            edgecolors="white",
            label="True TX",
        )
        ax.plot([tx_x_true, tx_x_est], [tx_y_true, tx_y_est], "w--", lw=1.2, alpha=0.8)

    ax.scatter(
        [tx_x_est],
        [tx_y_est],
        marker="x",
        s=110,
        c="cyan",
        linewidths=2.0,
        label="MF estimate",
    )

    error_str = "" if xy_error_m is None else f" | XY error {xy_error_m:.2f} m"
    ax.set_title(title + error_str)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(float(grid_x.min()), float(grid_x.max()))
    ax.set_ylim(float(grid_y.min()), float(grid_y.max()))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.colorbar(im, ax=ax, label="|matched-filter image|", fraction=0.02, pad=0.04)
    plt.show()



def main() -> int:
    args = parse_args()

    try:
        csi = load_csi_utils()

        # Load dataset
        ds, dataset_path = csi.open_dataset(
            experiment_id=args.experiment_id,
            dataset_path=args.dataset_path,
        )

        try:
            # Get available cycles
            available_cycles = csi.available_cycle_ids(ds, args.experiment_id)
            if int(args.cycle_id) not in set(available_cycles.astype(int).tolist()):
                raise ValueError(
                    f"Cycle {args.cycle_id} has no CSI for experiment {args.experiment_id}."
                )

            # Get selected cycle position
            selected_position = csi.cycle_position(ds, args.experiment_id, int(args.cycle_id))

            # Determine TX height
            if args.tx_height_m is None:
                if not selected_position["position_available"] or selected_position["rover_z"] is None:
                    raise ValueError(
                        "Selected cycle has no finite rover_z. Pass --tx-height-m to specify height."
                    )
                tx_height_m = float(selected_position["rover_z"])
            else:
                tx_height_m = float(args.tx_height_m)

            # Load antenna positions
            antenna_positions = csi.load_antenna_positions()

            # Extract measurement
            antenna_xyz, y = extract_measurement(
                ds,
                csi,
                args.experiment_id,
                int(args.cycle_id),
                antenna_positions,
            )

            # Filter out hosts with missing antenna coords
            finite_mask = np.isfinite(antenna_xyz).all(axis=1)
            antenna_xyz = antenna_xyz[finite_mask]
            y = y[finite_mask]

            # Build search grid
            grid_x, grid_y = build_search_grid()

            wavelength_m = C0 / float(args.frequency_hz)
            print(f"Loaded dataset: {dataset_path}")
            print(f"Experiment: {args.experiment_id} | Cycle: {int(args.cycle_id)}")
            print(f"Frequency: {args.frequency_hz / 1e6:.1f} MHz | Wavelength: {wavelength_m * 100:.2f} cm")
            print(f"TX height: {tx_height_m:.3f} m | Path-loss exponent: {float(args.path_loss_exponent):.2f}")
            print(f"Hosts used: {y.size} | Grid size: {grid_x.size} pixels\n")

            # --- MEASUREMENT ---
            print("Processing real measurement...")
            image_meas = matched_filter_image(
                y,
                antenna_xyz,
                grid_x,
                grid_y,
                tx_z_m=tx_height_m,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
            )
            tx_x_meas, tx_y_meas, _, peak_mag_meas = estimate_xy_from_image(grid_x, grid_y, image_meas)

            true_xy = None
            if selected_position["position_available"] and selected_position["rover_x"] is not None:
                true_xy = (float(selected_position["rover_x"]), float(selected_position["rover_y"]))
                error_meas = np.hypot(tx_x_meas - true_xy[0], tx_y_meas - true_xy[1])
                print(f"Estimated XY: ({tx_x_meas:.3f}, {tx_y_meas:.3f}) m")
                print(f"True XY:      ({true_xy[0]:.3f}, {true_xy[1]:.3f}) m")
                print(f"XY error:     {error_meas:.3f} m")
            else:
                print(f"Estimated XY: ({tx_x_meas:.3f}, {tx_y_meas:.3f}) m")

            plot_image(
                grid_x,
                grid_y,
                image_meas,
                antenna_xyz,
                tx_x_meas,
                tx_y_meas,
                f"Real measurement | {args.experiment_id} cycle {int(args.cycle_id)}",
                true_xy_m=true_xy,
                xy_error_m=float(np.hypot(tx_x_meas - true_xy[0], tx_y_meas - true_xy[1])) if true_xy else None,
            )

            # --- SIMULATED (noiseless) ---
            print("\nProcessing simulated channel (noiseless)...")
            sim_tx_x, sim_tx_y = true_xy if true_xy else (tx_x_meas, tx_y_meas)
            sim_y = simulate_channel(
                sim_tx_x,
                sim_tx_y,
                tx_height_m,
                antenna_xyz,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
                complex_gain=1.0,
                noise_std=0.0,
            )
            image_sim = matched_filter_image(
                sim_y,
                antenna_xyz,
                grid_x,
                grid_y,
                tx_z_m=tx_height_m,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
            )
            tx_x_sim, tx_y_sim, _, _ = estimate_xy_from_image(grid_x, grid_y, image_sim)
            error_sim = np.hypot(tx_x_sim - sim_tx_x, tx_y_sim - sim_tx_y)
            print(f"Simulated TX: ({sim_tx_x:.3f}, {sim_tx_y:.3f}) m")
            print(f"Estimated XY: ({tx_x_sim:.3f}, {tx_y_sim:.3f}) m")
            print(f"XY error:     {error_sim:.3f} m")

            plot_image(
                grid_x,
                grid_y,
                image_sim,
                antenna_xyz,
                tx_x_sim,
                tx_y_sim,
                f"Simulated (noiseless) | {args.experiment_id}",
                true_xy_m=(sim_tx_x, sim_tx_y),
                xy_error_m=float(error_sim),
            )

            # --- SIMULATED (with noise) ---
            print("\nProcessing simulated channel (with noise)...")
            rng = np.random.default_rng(7)
            noisy_sim_y = simulate_channel(
                sim_tx_x,
                sim_tx_y,
                tx_height_m,
                antenna_xyz,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
                complex_gain=1.0,
                noise_std=1.0,
                rng=rng,
            )
            image_noisy = matched_filter_image(
                noisy_sim_y,
                antenna_xyz,
                grid_x,
                grid_y,
                tx_z_m=tx_height_m,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
            )
            tx_x_noisy, tx_y_noisy, _, _ = estimate_xy_from_image(grid_x, grid_y, image_noisy)
            error_noisy = np.hypot(tx_x_noisy - sim_tx_x, tx_y_noisy - sim_tx_y)
            print(f"Simulated TX: ({sim_tx_x:.3f}, {sim_tx_y:.3f}) m (noise_std=1.0)")
            print(f"Estimated XY: ({tx_x_noisy:.3f}, {tx_y_noisy:.3f}) m")
            print(f"XY error:     {error_noisy:.3f} m")

            plot_image(
                grid_x,
                grid_y,
                image_noisy,
                antenna_xyz,
                tx_x_noisy,
                tx_y_noisy,
                f"Simulated (noisy) | {args.experiment_id}",
                true_xy_m=(sim_tx_x, sim_tx_y),
                xy_error_m=float(error_noisy),
            )

        finally:
            ds.close()

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
