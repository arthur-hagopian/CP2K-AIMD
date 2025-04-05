#!/usr/bin/env python3
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 04/2025

import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data

# -------------------------------
# CONSTANTS
# -------------------------------
AU_TO_EV = 27.2113838565563  # Hartree to eV


# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def extract_num(filename):
    """
    Extract numeric suffix from a filename for proper sorting.
    """
    match = re.search(r'(\d+)(?=\.cube$)', filename)
    return int(match.group(1)) if match else float('inf')


def read_and_process_cube(filename, electrode_surface, wat_lower, wat_upper, show_plot=False):
    """
    Reads cube file, averages over XY, computes bulk water potential in specified Z-range.
    If show_plot is True, plots the potential profile.
    """
    print(f"Processing: {filename}")
    try:
        data, atoms = read_cube_data(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None

    data_ev = data * AU_TO_EV
    avg_z_profile = np.mean(data_ev, axis=(0, 1))

    try:
        z_cell = atoms.get_cell()[2][2]
    except Exception as e:
        print(f"Error extracting cell from {filename}: {e}")
        return None, None, None

    nz = data.shape[2]
    z_axis = np.linspace(0, z_cell, nz)

    z_min = electrode_surface + wat_lower
    z_max = electrode_surface + wat_upper

    indices = np.where((z_axis >= z_min) & (z_axis <= z_max))[0]
    if indices.size == 0:
        print(f"Warning: no points found in water region for {filename}.")
        bulk_potential = np.nan
    else:
        bulk_potential = np.mean(avg_z_profile[indices])
        print(f"Bulk water potential (eV): {bulk_potential:.6f}")

    if show_plot:
        plot_profile(z_axis, avg_z_profile, z_min, z_max, os.path.basename(filename))

    return bulk_potential, z_axis, avg_z_profile


def plot_profile(z, potential, zmin, zmax, title):
    """
    Plot XY-averaged potential vs. z with water bulk limits highlighted.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(z, potential, label="XY-Averaged Potential")
    plt.axvline(zmin, linestyle="--", color="red", label="Water region")
    plt.axvline(zmax, linestyle="--", color="red")
    plt.xlabel("z-axis (Angstrom)")
    plt.ylabel("Potential (eV)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bulk_potential_vs_time(timesteps, potentials):
    """
    Plot water bulk potential vs. time and show standard error of the mean on the plot.
    """
    potentials = np.array(potentials)
    sem = np.std(potentials[~np.isnan(potentials)], ddof=1) / np.sqrt(np.sum(~np.isnan(potentials)))
    print(f"Standard error of mean bulk potential: {sem:.6f} eV")

    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, potentials, marker='o', linestyle='-', color='black')
    plt.xlabel("Time (fs)")
    plt.ylabel("Water Bulk Potential (eV)")
    plt.title("Water Bulk Potential vs. Time")
    plt.grid(True)
    plt.text(0.99, 0.98, f"SEM: {sem:.6f} eV", ha='right', va='top',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    plt.tight_layout()
    plt.show()


def save_results(timesteps, potentials, output_filename, wat_limits):
    """
    Save (time, potential) data to a file with informative header.
    """
    print(f"Writing output to: {output_filename}")
    header = f"TimeStep_fs    WaterBulkPotential_eV | Water region: {wat_limits[0]}–{wat_limits[1]} Å"
    try:
        np.savetxt(output_filename,
                   np.column_stack((timesteps, potentials)),
                   fmt='%12.2f %20.10f',
                   header=header)
    except Exception as e:
        print(f"Error writing to file: {e}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract water bulk potential from CP2K cube files.")
    parser.add_argument("-p", "--pattern", type=str, default="*hartree*.cube", help="Glob pattern (default: '*hartree*.cube')")
    parser.add_argument("--electrode-surface", type=float, default=7.0, help="Electrode surface position (Å)")
    parser.add_argument("--wat-lower-limit", type=float, default=5.0, help="Lower bound of water region (Å)")
    parser.add_argument("--wat-upper-limit", type=float, default=17.0, help="Upper bound of water region (Å)")
    parser.add_argument("--time-step", type=float, default=50.0, help="Timestep spacing in fs (default: 50 fs)")
    parser.add_argument("--plot", action="store_true", help="If set, plot first profile and final summary.")
    parser.add_argument("-o", "--output", type=str, default="water_bulk_potentials.dat", help="Output file")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern), key=extract_num)
    if not files:
        print(f"No files match pattern: {args.pattern}")
        return

    print(f"Found {len(files)} file(s). Starting analysis...")

    bulk_potentials = []
    plot_first_profile = True

    for f in files:
        bulk_potential, z_axis, avg_profile = read_and_process_cube(
            f,
            args.electrode_surface,
            args.wat_lower_limit,
            args.wat_upper_limit,
            show_plot=args.plot and plot_first_profile
        )
        if plot_first_profile:
            plot_first_profile = False
        bulk_potentials.append(np.nan if bulk_potential is None else bulk_potential)

    timesteps = np.arange(1, len(files) + 1) * args.time_step
    save_results(timesteps, bulk_potentials, args.output, (args.wat_lower_limit, args.wat_upper_limit))

    if args.plot:
        plot_bulk_potential_vs_time(timesteps, bulk_potentials)

    print("All processing completed.")


if __name__ == '__main__':
    main()

