#!/usr/bin/env python3
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 02/2025

"""
Script to compute the average density profile of a given element from an AIMD .xyz trajectory.

For each frame between the specified start and stop indices, the script:
  - Reads the .xyz frame (assuming standard .xyz format: first line = number of atoms,
    second line = comment, followed by one line per atom: "Element  x  y  z").
  - Extracts the z coordinates of atoms matching the specified element.
  - Bins the z coordinates into a user-defined number of bins along the z axis (using the
    simulation cell dimension in z provided by the user or a config file).
After processing all frames, the average number of atoms per bin is converted into a density in g/cm³. For elements with no defined atomic mass, the density is computed in arbitrary units (a.u.) as the count per volume.
The resulting density profile (bin center vs. density) is saved to an output file and plotted.
If --surface-position-avg is provided, the plotted z axis will be shifted so that the given value is 0.
"""

import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt

def parse_xyz(filename, start, stop):
    """
    Generator function to yield frames from an .xyz file.
    Each frame is returned as a list of tuples: (element, x, y, z).
    Only frames with indices in the range [start, stop] are yielded.
    """
    with open(filename, 'r') as f:
        frame_index = 0
        while True:
            line = f.readline()
            if not line:
                break  # End of file
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            # Skip comment line
            comment = f.readline()
            frame = []
            for i in range(natoms):
                atom_line = f.readline().strip()
                if not atom_line:
                    continue
                parts = atom_line.split()
                if len(parts) < 4:
                    continue
                element = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    continue
                frame.append((element, x, y, z))
            if start <= frame_index < stop:
                yield frame
            frame_index += 1
            if frame_index >= stop:
                break

def main():
    parser = argparse.ArgumentParser(
        description="Compute the average density profile of a given element from an AIMD .xyz trajectory."
    )
    parser.add_argument("input", type=str, help="Input .xyz trajectory file.")
    parser.add_argument("--cell-x", type=float, default=None, help="Simulation cell dimension in x (Å).")
    parser.add_argument("--cell-y", type=float, default=None, help="Simulation cell dimension in y (Å).")
    parser.add_argument("--cell-z", type=float, default=None, help="Simulation cell dimension in z (Å).")
    parser.add_argument("--cell-config", type=str, default=None,
                        help="Path to an INI configuration file with default cell dimensions (requires [cell] section with cell_x, cell_y, cell_z).")
    parser.add_argument("--element", type=str, default="O", help="Element symbol to analyze (default: O).")
    parser.add_argument("--start", type=int, default=0, help="Frame index at which to start averaging (default: 0).")
    parser.add_argument("--stop", type=int, default=15000, help="Frame index at which to stop averaging (default: 15000).")
    parser.add_argument("--nbins", type=int, default=300, help="Number of bins along z (default: 300).")
    parser.add_argument("--surface-position-avg", type=float, default=None,
                        help="If provided, shift the z-axis for the plot so that this value corresponds to zero (e.g., 7).")
    parser.add_argument("--plot", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    # Load cell dimensions from configuration file if provided and not already set
    if args.cell_config:
        config = configparser.ConfigParser()
        config.read(args.cell_config)
        if "cell" not in config:
            parser.error("Cell configuration file must contain a [cell] section.")
        if args.cell_x is None:
            args.cell_x = float(config["cell"]["cell_x"])
        if args.cell_y is None:
            args.cell_y = float(config["cell"]["cell_y"])
        if args.cell_z is None:
            args.cell_z = float(config["cell"]["cell_z"])

    # Ensure that cell dimensions are provided
    if args.cell_x is None or args.cell_y is None or args.cell_z is None:
        parser.error("Cell dimensions must be provided either via command-line or via a configuration file.")

    # Define bin edges along z (assume lower bound = 0 and upper bound = cell_z)
    bins = np.linspace(0, args.cell_z, args.nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # Define atomic masses (in grams) if available.
    atomic_masses = {
        "O": 15.999 * 1.66053906660e-24,
        "H": 1.008 * 1.66053906660e-24,
        "Li": 6.94 * 1.66053906660e-24,
        "Na": 22.98976928 * 1.66053906660e-24,
        "K": 39.0983 * 1.66053906660e-24,
        "Cs": 132.90545 * 1.66053906660e-24
    }
    if args.element in atomic_masses:
        use_arbitrary_units = False
        atom_mass = atomic_masses[args.element]
        density_unit = "g/cm^3"
    else:
        use_arbitrary_units = True
        print(f"Atomic mass for element {args.element} is not defined. Computing density in arbitrary units (a.u.).")
        density_unit = "a.u."

    # Initialize accumulator for histogram counts (one count per bin)
    histogram_accum = np.zeros(args.nbins)
    frame_count = 0

    # Process each frame in the specified range
    for frame in parse_xyz(args.input, args.start, args.stop):
        # Extract z coordinates for atoms of the target element
        z_coords = [z for (el, x, y, z) in frame if el == args.element]
        if not z_coords:
            continue
        counts, _ = np.histogram(z_coords, bins=bins)
        histogram_accum += counts
        frame_count += 1

    if frame_count == 0:
        print("No frames processed or no atoms of the specified element found.")
        return

    # Average counts per bin over all processed frames
    avg_counts = histogram_accum / frame_count

    # Calculate the volume of a single bin in cm^3.
    # Cell dimensions are given in Å; 1 Å³ = 1e-24 cm³.
    volume_bin_A3 = args.cell_x * args.cell_y * bin_width
    volume_bin_cm3 = volume_bin_A3 * 1e-24

    # Compute density:
    if not use_arbitrary_units:
        density = (avg_counts * atom_mass) / volume_bin_cm3
    else:
        density = avg_counts / volume_bin_cm3

    # Save the density profile to an output file
    output_file = f"{args.element}_density.dat"
    header_str = f"Z-axis (angstrom)    Density ({density_unit})    (Averaged over frames {args.start} to {args.stop})"
    data_to_save = np.column_stack((bin_centers, density))
    np.savetxt(output_file, data_to_save, fmt="%.6f", header=header_str)
    print(f"Density profile saved to {output_file}.")

    # Prepare data for plotting.
    # If a surface position is provided, shift the bin centers for the plot.
    if args.surface_position_avg is not None:
        plot_bin_centers = bin_centers - args.surface_position_avg
        xlabel = f"z (Å) relative to surface at {args.surface_position_avg} Å"
    else:
        plot_bin_centers = bin_centers
        xlabel = "z (Å)"

    # Plot the density profile
    plt.figure(figsize=(8, 6))
    plt.plot(plot_bin_centers, density, label=f"{args.element} Density")
    plt.xlabel(xlabel)
    plt.ylabel(f"Density ({density_unit})")
    plt.title(f"Average {args.element} Density Profile")
    plt.legend()
    plt.grid(True)
    if args.plot:
        plt.show()
    else:
        plt.savefig(f"{args.element}_density.png")
        print(f"Plot saved as {args.element}_density.png.")

if __name__ == "__main__":
    main()

