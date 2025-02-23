#!/usr/bin/env python3
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 02/2025
# Description: Script to process .cube files and compute the average water bulk potential.
#              For each cube file, the script reads the potential (in au), converts it to eV,
#              averages the potential over the xy coordinates, and computes the average value in the 
#              water bulk region defined between ELECTRODE_SURFACE+WAT_LOWER_LIMIT and 
#              ELECTRODE_SURFACE+WAT_UPPER_LIMIT. It then plots the full xy-averaged profile vs. z and 
#              writes the water bulk average (with its corresponding time-step) to an output file.
#
# Usage:
#   Run the script without arguments to process files matching "*hartree*.cube"
#   Use -h for help and to see available options.
#
#   Example:
#       ./process_cube_files.py --pattern "*hartree*.cube" --plot
#
#   Make sure to use the correct following parameters:
#       ELECTRODE_SURFACE, WAT_LOWER_LIMIT, WAT_UPPER_LIMIT, time-step, and the output filename.
#
import os
import glob
import logging
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data

# Constant for unit conversion: Hartree to eV
AU_TO_EV = 27.2113838565563

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

def extract_num(filename):
    """
    Extracts the numerical value immediately preceding the '.cube' extension in the filename.
    If no such number is found, returns infinity so that such files are sorted at the end.
    """
    match = re.search(r'(\d+)(?=\.cube$)', filename)
    return int(match.group(1)) if match else float('inf')

def process_cube_file(filename, show_plot, electrode_surface, wat_lower_limit, wat_upper_limit):
    """
    Reads a cube file, converts potential values to eV, averages over xy,
    computes the water bulk average between (electrode_surface+wat_lower_limit)
    and (electrode_surface+wat_upper_limit), defines a z-axis, and plots the averaged 
    potential as a function of z.

    Returns:
        water_bulk_avg: The average potential (in eV) in the water bulk region.
    """
    logging.info(f"Processing file: {filename}")
    try:
        # read_cube_data returns (data, atoms) for these cube files.
        data, atoms = read_cube_data(filename)
    except Exception as e:
        logging.error(f"Error reading cube file {filename}: {e}")
        return None

    # Convert potential data from atomic units (au) to eV
    data_ev = data * AU_TO_EV

    # Average the data over the x and y dimensions to obtain a 1D profile along z
    xy_potential_avg = np.mean(data_ev, axis=(0, 1))

    # Define the z-axis using the cell information from the atoms object.
    try:
        z_cell = atoms.get_cell()[2][2]
    except Exception as e:
        logging.error(f"Error obtaining cell information from {filename}: {e}")
        return None

    z_grid_points = data.shape[2]
    z_axis = [z / z_grid_points * z_cell for z in range(len(xy_potential_avg))]

    # Determine water bulk region limits
    water_bulk_region_min = electrode_surface + wat_lower_limit
    water_bulk_region_max = electrode_surface + wat_upper_limit

    # Find indices in the z-axis that lie within the water bulk region
    z_axis_arr = np.array(z_axis)
    indices = np.where((z_axis_arr >= water_bulk_region_min) & (z_axis_arr <= water_bulk_region_max))[0]
    if indices.size == 0:
        logging.warning(f"No grid points found in water bulk region for {filename}.")
        water_bulk_avg = np.nan
    else:
        water_bulk_avg = np.mean(xy_potential_avg[indices])
        logging.info(f"Water bulk average for {filename}: {water_bulk_avg:.6f} eV")

    # Create the plot of the full xy-averaged potential vs. z
    fig, ax = plt.subplots()
    ax.plot(z_axis, xy_potential_avg, label="XY-Averaged Potential")
    ax.axvline(x=water_bulk_region_min, color='r', linestyle='--', label="Water bulk limits")
    ax.axvline(x=water_bulk_region_max, color='r', linestyle='--')
    ax.set_xlabel("z-axis (Angstrom)")
    ax.set_ylabel("Potential (eV)")
    ax.set_title(f"XY-Averaged Potential for {os.path.basename(filename)}")
    ax.legend()
    ax.grid(True)

    # Do not save the figure; only display if requested.
    if show_plot:
        plt.show()
    plt.close(fig)

    return water_bulk_avg

def main():
    parser = argparse.ArgumentParser(
        description="Process .cube files with 'hartree' in their names to extract "
                    "the xy-averaged potential (converted from au to eV) and compute the average "
                    "potential in the water bulk region. The water bulk region is defined as the z-range "
                    "between (ELECTRODE_SURFACE + WAT_LOWER_LIMIT) and (ELECTRODE_SURFACE + WAT_UPPER_LIMIT)."
    )
    parser.add_argument(
        "-p", "--pattern", type=str, default="*hartree*.cube",
        help="Glob pattern to match cube files (default: '*hartree*.cube')"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="If set, display the plots interactively."
    )
    parser.add_argument(
        "--electrode-surface", type=float, default=7.0,
        help="Position of the electrode surface (default: 7.0)"
    )
    parser.add_argument(
        "--wat-lower-limit", type=float, default=5.0,
        help="Lower limit for the water region relative to the electrode surface (default: 5.0)"
    )
    parser.add_argument(
        "--wat-upper-limit", type=float, default=17.0,
        help="Upper limit for the water region relative to the electrode surface (default: 17.0)"
    )
    parser.add_argument(
        "--time-step", type=float, default=50,
        help="Time step interval to assign to each cube file (default: 50)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="water_bulk_potentials.dat",
        help="Output filename for water bulk potentials (default: water_bulk_potentials.dat)"
    )
    args = parser.parse_args()

    # Find and sort all cube files matching the given pattern in the current directory
    cube_files = sorted(glob.glob(args.pattern), key=extract_num)
    if not cube_files:
        logging.error(f"No cube files matching the pattern '{args.pattern}' found in the current directory.")
        sys.exit(1)
    logging.info(f"Found {len(cube_files)} file(s) matching the pattern '{args.pattern}'.")

    # List to store water bulk averages for each file
    water_bulk_averages = []

    # Process each cube file
    for i, cube_file in enumerate(cube_files, start=1):
        water_bulk_avg = process_cube_file(
            cube_file,
            args.plot,
            args.electrode_surface,
            args.wat_lower_limit,
            args.wat_upper_limit
        )
        if water_bulk_avg is not None:
            water_bulk_averages.append(water_bulk_avg)
        else:
            water_bulk_averages.append(np.nan)

    # Assign time steps: first file at args.time_step, second at 2*args.time_step, etc.
    time_steps = np.arange(args.time_step, (len(cube_files)+1) * args.time_step, args.time_step)

    # Save the water bulk averages to the output file with two columns: time and potential
    try:
        np.savetxt(args.output,
                   np.column_stack((time_steps, water_bulk_averages)),
                   fmt=("%.2f", "%.8f"),
                   delimiter="    ",
                   header=f"TimeStep    WaterBulkPotential (eV) | Bulk Region : {args.wat_lower_limit} - {args.wat_upper_limit}")
        logging.info(f"Water bulk potentials saved to: {args.output}")
    except Exception as e:
        logging.error(f"Error saving water bulk potentials: {e}")

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()

