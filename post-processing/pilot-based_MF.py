#!/usr/bin/env python3
"""RF backprojection matched filter from tutorial_rf_bp.ipynb with measurement and simulation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import traceback
from typing import Any
import pandas as pd 

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
        default="EXP003",
        help="Experiment ID to localize. Default: EXP003.",
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
    x_bounds_m: tuple[float, float] = (0.0, 7.99),
    y_bounds_m: tuple[float, float] = (0.0, 3.99),
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
    pilot_coefficients: np.ndarray,
    tx_z_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Compute channel image map over grid."""
    # print pilot_coefficients for debugging
    #print(f"pilot_coefficients: {pilot_coefficients}")
    wavelength = C0 / float(frequency_hz)
    k = 2.0 * np.pi / wavelength

    dx = grid_x[..., None] - antenna_xyz[:, 0][None, None, :]
    # print antenna_xyz[:, 0]
    #print(f"antenna_xyz[:, 0]: {antenna_xyz[:, 0]}")
    #print(f"antenna_xyz[:, 0] shape: {antenna_xyz[:, 0].shape}")
    dy = grid_y[..., None] - antenna_xyz[:, 1][None, None, :]
    dz = float(tx_z_m) - antenna_xyz[:, 2][None, None, :]
    #print(f"dx shape: {dx.shape} | dy shape: {dy.shape} | dz shape: {dz.shape}")
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    #print(f"distance shape: {distance.shape}")
    distance = np.maximum(distance, 1e-6)
    amplitude = 1.0 / np.power(distance, float(path_loss_exponent))
    return amplitude * np.exp(-1j * k * distance) # change here                       ############

def matched_filter_image(
    y: np.ndarray,
    antenna_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    pilot_coefficients: np.ndarray, 
    tx_z_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Compute matched-filter image: sum(conj(a(q)) * y)."""
    image_map = channel_image_map(
        antenna_xyz=antenna_xyz,
        grid_x=grid_x,
        grid_y=grid_y,
        pilot_coefficients=pilot_coefficients,
        tx_z_m=tx_z_m,
        frequency_hz=frequency_hz,
        path_loss_exponent=path_loss_exponent,
    )
    # y=y*pilot_coefficients element-wise multiplication to apply pilot coefficients to measurements
    y = y * pilot_coefficients
    MF = np.sum(np.conjugate(image_map) * y[None, None, :], axis=2)
    MF = MF ** 2.0
    return MF

def pilot_estimation(
    y: np.ndarray,
    antenna_xyz: np.ndarray,
    true_xy_pilot: tuple[float, float],
    tx_z_m: float,
    frequency_hz: float,
    path_loss_exponent: float = 1.0,
) -> np.ndarray:
    """Estimate pilot coefficients from measurement."""
    wavelength = C0 / float(frequency_hz)
    k = 2.0 * np.pi / wavelength

    dx_true = true_xy_pilot[0] - antenna_xyz[:, 0]
    #rint(f"dx shape: {dx.shape}")
    dy_true = true_xy_pilot[1] - antenna_xyz[:, 1]
    dz_true = float(tx_z_m) - antenna_xyz[:, 2]
    print(f"dx shape: {dx_true.shape} | dy shape: {dy_true.shape} | dz shape: {dz_true.shape}")
    distance = np.sqrt(dx_true**2 + dy_true**2 + dz_true**2)
    print(f"distance shape: {distance.shape}")
    distance = np.maximum(distance, 1e-6)
    amplitude = 1.0 / np.power(distance, float(path_loss_exponent))
    h = amplitude * np.exp(-1j * k * distance)
    pilot_coeff = np.divide(h, y)
    # normalize pilot coefficients to have unit magnitude
    pilot_coeff = pilot_coeff / np.abs(pilot_coeff)
    return pilot_coeff

def estimate_xy_from_image(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    image: np.ndarray,
    acostic_pred_x: float,
    acostic_pred_y: float,
) -> tuple[float, float, complex, float, float, float]:
    """Estimate TX position from matched-filter image."""
    magnitude = np.abs(image)
    # rf only
    best_idx_rf = int(np.nanargmax(magnitude))
    iy_rf, ix_rf = np.unravel_index(best_idx_rf, magnitude.shape)
    # limit search to a 0.3259 m radius around acoustic prediction
    if acostic_pred_x is not None and acostic_pred_y is not None:
        radius_m = 0.3259/2
        mask = (grid_x - acostic_pred_x) ** 2 + (grid_y - acostic_pred_y) ** 2 <= radius_m ** 2
        magnitude = np.where(mask, magnitude, 0.0)
        #print(f"Applied circular mask with radius {radius_m} m around acoustic prediction ({acostic_pred_x:.3f}, {acostic_pred_y:.3f}) m")
    best_idx = int(np.nanargmax(magnitude))
    iy, ix = np.unravel_index(best_idx, magnitude.shape)
    return (
        float(grid_x[iy, ix]),
        float(grid_y[iy, ix]),
        complex(image[iy, ix]),
        float(magnitude[iy, ix]),
        float(grid_x[iy_rf, ix_rf]),
        float(grid_y[iy_rf, ix_rf]),
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
    if snapshot is None:
        raise ValueError(f"csi.extract_csi_snapshot returned None for cycle {cycle_id}")
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
    # element wise normalize y to have unit magnitude
    #y = y / np.abs(y)
    return antenna_xyz, y


def plot_image(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    image: np.ndarray,
    antenna_xyz: np.ndarray,
    tx_x_est: float,
    tx_y_est: float,
    acostic_pred_x: float,
    acostic_pred_y: float,
    title: str,
    true_xy_m: tuple[float, float] | None = None,
    xy_error_m: float | None = None,
    acoustic_error_m: float | None = None,
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
        label="RF estimate",
    )

    ax.scatter(
        [acostic_pred_x],
        [acostic_pred_y],
        marker="s",
        s=50,
        c="magenta",
        linewidths=2.0,
        label="Acoustic prediction",
    )

    error_str = "" if xy_error_m is None else f" | rf error {xy_error_m:.2f}, acoustic error {acoustic_error_m:.2f} m"
    ax.set_title(title + error_str)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(float(grid_x.min()), float(grid_x.max()))
    ax.set_ylim(float(grid_y.min()), float(grid_y.max()))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.colorbar(im, ax=ax, label="|matched-filter image|", fraction=0.046, pad=0.04)
    plt.show()



def main() -> int:
    args = parse_args()
    try:
        csi = load_csi_utils()

        # Load dataset
        args.experiment_id = "EXP007" 
        ds, dataset_path = csi.open_dataset(
            experiment_id=args.experiment_id,
            dataset_path=args.dataset_path,
        )

        error_fusion = []
        error_acoustic = []
        error_rf = []

        printresults= 0
        plotimg = 0
        # Get available cycles
        available_cycles = csi.available_cycle_ids(ds, args.experiment_id)
        #print(f"Available cycles for experiment {args.experiment_id}: {available_cycles.astype(int).tolist()}")
        if int(args.cycle_id) not in set(available_cycles.astype(int).tolist()):
            raise ValueError(
                f"Cycle {args.cycle_id} has no CSI for experiment {args.experiment_id}."
            )

        # select cycle 1 for pilot estimation and height estimation, then cycle 100 for localization 
        args.cycle_id = int(1) # pilot cycle
        selected_position = csi.cycle_position(ds, args.experiment_id, int(args.cycle_id))
        if selected_position is None:
            raise ValueError(f"csi.cycle_position returned None for pilot cycle {int(args.cycle_id)}")
        print(f"Selected cycle position: {selected_position}")
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
        print(f"PILOT Experiment: {args.experiment_id} | Cycle: {int(args.cycle_id)}")
        print(f"Frequency: {args.frequency_hz / 1e6:.1f} MHz | Wavelength: {wavelength_m * 100:.2f} cm")
        print(f"TX height: {tx_height_m:.3f} m | Path-loss exponent: {float(args.path_loss_exponent):.2f}")
        print(f"Hosts used: {y.size} | Grid size: {grid_x.size} pixels\n")

        # check number of measurements in antenna_x for all cycles to confirm that cycle 1 has enough measurements for pilot estimation
        #for cycle in available_cycles.astype(int).tolist():
        #    num_measurements_ant_x = csi.cycle_position(ds, args.experiment_id, cycle)['csi_host_count']
        #    print(f"Number of measurements in cycle {cycle}: {num_measurements_ant_x}")
        # --- MEASUREMENT ---
        true_xy_pilot = (float(selected_position["rover_x"]), float(selected_position["rover_y"]))
        print("Processing pilot measurement and collect pilot coefficients...")
        pilot_coefficients = pilot_estimation(
            y, 
            antenna_xyz, 
            true_xy_pilot,
            tx_z_m=tx_height_m, 
            frequency_hz=float(args.frequency_hz), 
            path_loss_exponent=float(args.path_loss_exponent)
        )
        

        print("Processing real measurement...")
        # read a csv file in this directory called :Localization_Predictions_SearchBack_T050.csv to get the estimated positions for arbitrary experiment_number and cycle_number
        acoustic_predictions = pd.read_csv("Localization_Predictions_SearchBack_T050.csv")
        
        # create loop for all cycle ids in available_cycles to process each measurement cycle and plot results, for now just process cycle 100
        for cycle in available_cycles.astype(int).tolist(): #range(1,10): # 
            pos_info = csi.cycle_position(ds, args.experiment_id, cycle)
            if pos_info is None:
                print(f"Skipping cycle {cycle}: no position info available.", file=sys.stderr)
                continue
            num_measurements_ant_x = pos_info['csi_host_count']
            if num_measurements_ant_x < 42: # skip cycles with too few measurements for localization
                print(f"Skipping cycle {cycle} with only {num_measurements_ant_x} measurements.")
                continue
            print(f"\nProcessing cycle {cycle}...")

            args.cycle_id = int(cycle) # real measurement cycle
            selected_position = csi.cycle_position(ds, args.experiment_id, int(args.cycle_id))
            if selected_position is None:
                print(f"Skipping cycle {cycle}: selected_position is None.", file=sys.stderr)
                continue
            # find a line in the csv where experiment_id column matches args.experiment_id and cycle_id column matches args.cycle_id and print the predicted x and y positions from columns pred_x and pred_y
            matching_row = acoustic_predictions[(acoustic_predictions['experiment_number'] == args.experiment_id) & (acoustic_predictions['cycle_number'] == args.cycle_id)]
            if not matching_row.empty:
                acostic_pred_x = matching_row['x_3d_m'].iloc[0]
                acostic_pred_y = matching_row['y_3d_m'].iloc[0]
                #print(f"Acoustic prediction from CSV for Experiment {args.experiment_id} Cycle {int(args.cycle_id)}: ({acostic_pred_x:.3f}, {acostic_pred_y:.3f}) m")
            
            antenna_xyz, y = extract_measurement(
                ds,
                csi,
                args.experiment_id,
                int(args.cycle_id),
                antenna_positions,
            )
            
            image_meas = matched_filter_image(
                y,
                antenna_xyz,
                grid_x,
                grid_y,
                pilot_coefficients,
                tx_z_m=tx_height_m,
                frequency_hz=float(args.frequency_hz),
                path_loss_exponent=float(args.path_loss_exponent),
            )
            tx_x_est, tx_y_est, _, peak_mag_meas, tx_x_est_rf, tx_y_est_rf = estimate_xy_from_image(grid_x, grid_y, image_meas, acostic_pred_x, acostic_pred_y)

            true_xy = None 
            if selected_position["position_available"] and selected_position["rover_x"] is not None:
                true_xy = (float(selected_position["rover_x"]), float(selected_position["rover_y"]))
                error_fusion_inst = np.sqrt((tx_x_est - true_xy[0]) ** 2 + (tx_y_est - true_xy[1]) ** 2)
                # print true and estimated positions and error for this cycle in one line
                print(f"True XY:      ({true_xy[0]:.3f}, {true_xy[1]:.3f}) m | Estimated fusion XY: ({tx_x_est:.3f}, {tx_y_est:.3f}) m | Estimated error fusion: {error_fusion_inst:.3f} m", end="")
                # append error to list for all cycles
                error_fusion.append(error_fusion_inst)
                error_rf_inst = np.sqrt((tx_x_est_rf - true_xy[0]) ** 2 + (tx_y_est_rf - true_xy[1]) ** 2)
                print(f" | Estimated RF-only XY: ({tx_x_est_rf:.3f}, {tx_y_est_rf:.3f}) m | Estimated error RF-only: {error_rf_inst:.3f} m", end="")
                error_rf.append(error_rf_inst)
                acoustic_error_inst = np.sqrt((acostic_pred_x - true_xy[0]) ** 2 + (acostic_pred_y - true_xy[1]) ** 2)
                #print acoustic prediction error for this cycle
                print(f" | Acoustic prediction XY: ({acostic_pred_x:.3f}, {acostic_pred_y:.3f}) m | Acoustic prediction error: {acoustic_error_inst:.3f} m")
                error_acoustic.append(acoustic_error_inst)
                if printresults==1:
                    print(f"Estimated XY: ({tx_x_est:.3f}, {tx_y_est:.3f}) m")
                    print(f"True XY:      ({true_xy[0]:.3f}, {true_xy[1]:.3f}) m")
                print(f"XY error fusion:     {error_fusion_inst:.3f} m")
                print(f"XY error acoustic:   {acoustic_error_inst:.3f} m")
            #plot results for this cycle including matched filter image and fusion estimated position, true position, and acoustic prediction
            if plotimg==1:
                plot_image(
                    grid_x,
                    grid_y,
                    image_meas,
                    antenna_xyz,
                    tx_x_est,
                    tx_y_est,
                    acostic_pred_x,
                    acostic_pred_y,
                    title=f"Experiment {args.experiment_id} Cycle {int(args.cycle_id)} Calibrated Matched-Filter Image",
                    true_xy_m=true_xy,
                    xy_error_m=error_fusion_inst if true_xy is not None else None,
                    acoustic_error_m=acoustic_error_inst if true_xy is not None else None,
                )
        # summary statistics        
        print(f"\nAverage XY fusion error over {len(error_fusion)} cycles: {np.mean(error_fusion):.3f} m")
        print(f"\nAverage XY acoustic error over {len(error_acoustic)} cycles: {np.mean(error_acoustic):.3f} m")
        print(f"\nAverage XY RF-only error over {len(error_rf)} cycles: {np.mean(error_rf):.3f} m")

        #plot empirical CDF by sorting the error_fusion, error_acoustic, and error_rf lists and plotting the sorted values against their percentile rank
        # chenge the x-axis to log scale to better visualize the differences at lower error values
        plt.figure(figsize=(8, 6))
        sorted_fusion = np.sort(error_fusion)
        sorted_acoustic = np.sort(error_acoustic)
        sorted_rf = np.sort(error_rf)
        plt.semilogx(sorted_fusion, np.arange(1, len(sorted_fusion) + 1) / len(sorted_fusion), label="Fusion")
        plt.semilogx(sorted_acoustic, np.arange(1, len(sorted_acoustic) + 1) / len(sorted_acoustic), label="Acoustic-only")
        plt.semilogx(sorted_rf, np.arange(1, len(sorted_rf) + 1) / len(sorted_rf), label="RF-only")
        plt.xlabel("Localization error [m]")
        plt.ylabel("Empirical CDF")
        # include mean in the title
        plt.title(f"Empirical CDF of Localization Error (Fusion mean: {np.mean(error_fusion):.3f} m, Acoustic-only mean: {np.mean(error_acoustic):.3f} m, RF-only mean: {np.mean(error_rf):.3f} m)")
        #include line for mean error for each method
        plt.axvline(np.mean(error_fusion), color='blue', linestyle='--', label=f'Fusion mean: {np.mean(error_fusion):.3f} m')
        plt.axvline(np.mean(error_acoustic), color='orange', linestyle='--', label=f'Acoustic-only mean: {np.mean(error_acoustic):.3f} m')
        plt.axvline(np.mean(error_rf), color='green', linestyle='--', label=f'RF-only mean: {np.mean(error_rf):.3f} m')
        plt.grid()
        plt.legend()
        plt.show()

    except Exception as exc:
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())