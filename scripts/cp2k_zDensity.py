#!/usr/bin/env python3
# Author: Arthur Hagopian
# Date: 04/2025

import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# DATA I/O FUNCTIONS
# -------------------------------
def parse_xyz(filename, start, stop):
    with open(filename, 'r') as f:
        frame_idx = 0
        while True:
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            f.readline()  # skip comment line
            frame = []
            for _ in range(natoms):
                atom_line = f.readline().strip()
                if not atom_line:
                    continue
                parts = atom_line.split()
                if len(parts) < 4:
                    print(f"  [Warning] Incomplete atom line: {atom_line}")
                    continue
                element = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                    frame.append((element, x, y, z))
                except ValueError:
                    continue
            if start <= frame_idx < stop:
                yield frame
            frame_idx += 1
            if frame_idx >= stop:
                break

def count_total_frames(filename):
    total = 0
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
                f.readline()
                for _ in range(natoms):
                    f.readline()
                total += 1
            except Exception:
                break
    return total


# -------------------------------
# SURFACE DETECTION
# -------------------------------
def compute_surface_z_position(xyz_path, start, stop, metal_list=["Cu", "Au", "Pt", "Pd"], top_percent=5):
    z_surf_all = []
    surface_atom_counts = []

    for frame in parse_xyz(xyz_path, start, stop):
        print(f"Frame has {len(frame)} atoms; first 5: {[el for el, *_ in frame[:5]]}")
        metal_z = [z for el, _, _, z in frame if el in metal_list]
        print(f"  → Found {len(metal_z)} metal atoms in this frame.")
        if not metal_z:
            continue
        z_sorted = np.sort(metal_z)
        n_surface = max(1, int(len(z_sorted) * top_percent / 100))
        z_top = z_sorted[-n_surface:]
        z_avg = np.mean(z_top)
        z_surf_all.append(z_avg)
        surface_atom_counts.append(n_surface)

    if not z_surf_all:
        print("No metal atoms found in frames.")
        return None

    print(f"Average number of surface atoms per frame: {np.mean(surface_atom_counts):.1f}")
    z_surf_mean = np.mean(z_surf_all)
    print(f"Estimated average electrode surface z-position: {z_surf_mean:.3f} Å")
    return z_surf_mean


# -------------------------------
# DENSITY ANALYSIS
# -------------------------------
def compute_density_profile(xyz_path, element, start, stop, bins, bin_width, cell_x, cell_y):
    atomic_masses = {
        "O": 15.999 * 1.66053906660e-24,
        "H": 1.008 * 1.66053906660e-24,
        "Li": 6.94 * 1.66053906660e-24,
        "Na": 22.98976928 * 1.66053906660e-24,
        "K": 39.0983 * 1.66053906660e-24,
        "Cs": 132.90545 * 1.66053906660e-24
    }

    if element in atomic_masses:
        atom_mass = atomic_masses[element]
        density_unit = "g/cm^3"
        use_au = False
    else:
        print(f"Warning: Atomic mass for {element} not defined. Using arbitrary units (a.u.).")
        density_unit = "a.u."
        use_au = True

    histogram = np.zeros(len(bins) - 1)
    n_frames = 0

    for frame in parse_xyz(xyz_path, start, stop):
        z_vals = [z for el, _, _, z in frame if el == element]
        if not z_vals:
            continue
        counts, _ = np.histogram(z_vals, bins=bins)
        histogram += counts
        n_frames += 1

    if n_frames == 0:
        print("No frames processed. Exiting.")
        return None, None, None

    avg_counts = histogram / n_frames
    bin_volume_cm3 = cell_x * cell_y * bin_width * 1e-24

    if use_au:
        density = avg_counts / bin_volume_cm3
    else:
        density = (avg_counts * atom_mass) / bin_volume_cm3

    return density, n_frames, density_unit


# -------------------------------
# PLOTTING AND OUTPUT
# -------------------------------
def plot_density_profile(z_centers, density, element, unit, shift=None, show=False):
    if shift is not None:
        z_centers = z_centers - shift
        xlabel = "Distance to surface (Å)"
    else:
        xlabel = "z (Å)"

    plt.figure(figsize=(8, 6))
    plt.fill_between(z_centers, density, color='skyblue', alpha=0.4)
    plt.plot(z_centers, density, color='blue', linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(f"$\\rho_{{{element}}}$ ({unit})")
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        filename = f"{element}_density.png"
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")


def save_density_data(z_centers, density, element, unit, start, stop):
    output_file = f"{element}_density.dat"
    header = f"Z (Å)    Density ({unit})    (Frames {start} to {stop})"
    np.savetxt(output_file, np.column_stack((z_centers, density)), fmt="%.6f", header=header)
    print(f"Density data saved to: {output_file}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute average z-density profile from CP2K trajectory.")
    parser.add_argument("input", type=str, nargs='?', default="trajectories.xyz", help="Input .xyz trajectory")
    parser.add_argument("--cell-config", type=str, default="cell.ini", help="INI config with [cell] section (default: cell.ini)")
    parser.add_argument("--element", type=str, default="O", help="Element to analyze (default: O)")
    parser.add_argument("--start", type=int, default=0, help="First frame (default: 0)")
    parser.add_argument("--stop", type=int, default=15000, help="Last frame (default: 15000)")
    parser.add_argument("--nbins", type=int, default=300, help="Number of bins along z (default: 300)")
    parser.add_argument("--plot", action="store_true", help="Show interactive plot")
    args = parser.parse_args()

    total_frames = count_total_frames(args.input)
    if args.start >= total_frames:
        print(f"Warning: start frame ({args.start}) >= total frames in file ({total_frames}). Exiting.")
        return
    if args.stop > total_frames:
        print(f"Warning: stop frame ({args.stop}) exceeds available frames ({total_frames}). Clamping to {total_frames}.")
        args.stop = total_frames

    config = configparser.ConfigParser()
    config.read(args.cell_config)
    try:
        cell_x = float(config["cell"]["cell_x"])
        cell_y = float(config["cell"]["cell_y"])
        cell_z = float(config["cell"]["cell_z"])
    except Exception as e:
        parser.error(f"Failed to read [cell] from {args.cell_config}: {e}")

    bins = np.linspace(0, cell_z, args.nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    print(f"Computing density for element: {args.element}")
    print(f"Using frames {args.start} to {args.stop}")
    print(f"Cell dimensions: {cell_x:.2f} x {cell_y:.2f} x {cell_z:.2f} Å")

    z_surface_avg = compute_surface_z_position(args.input, args.start, args.stop)

    density, n_frames, unit = compute_density_profile(
        args.input, args.element, args.start, args.stop, bins, bin_width, cell_x, cell_y
    )
    if density is None:
        return

    save_density_data(bin_centers, density, args.element, unit, args.start, args.stop)
    plot_density_profile(bin_centers, density, args.element, unit, shift=z_surface_avg, show=args.plot)
    print("Completed.")


if __name__ == '__main__':
    main()
