#!/usr/bin/env python3
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 02/2025

"""
Script to extract Fermi energies at every time step of an AIMD run from CP2K output file.
The resulting data (time step vs. Fermi energy) is saved to "fermi_energies.dat"
"""

import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
#import matplotlib_parameters

# Constants
AU_TO_EV = 27.2113838565563  # Hartree to eV conversion factor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def read_fermi_energies(filename, frequency_collect_fermi):
    """Reads Fermi energies from a given output file."""
    if not os.path.isfile(filename):
        logging.error(f"File not found: {filename}")
        return [], 0

    fermi_energies = []
    steps = 0
    fermi_pattern = re.compile(r".*Fermi.*")

    with open(filename, "r") as file:
        for line in file:
            if fermi_pattern.match(line):
                steps += 1
                if steps % frequency_collect_fermi == 0:
                    try:
                        fermi_energies.append(float(line.split()[-1]) * AU_TO_EV)
                    except ValueError as e:
                        logging.warning(f"Skipping line due to error: {line.strip()} - {e}")

    logging.info(f"Total simulation steps: {steps}")
    logging.info(f"Collected {len(fermi_energies)} Fermi energies")

    return fermi_energies


def save_fermi_energies(filename, steps, fermi_energies):
    """Saves Fermi energies and corresponding steps to a file with a header."""
    try:
        header_str = "Step    FermiEnergy (eV)"
        np.savetxt(filename,
                   np.column_stack((steps, fermi_energies)),
                   fmt=("%.2f", "%.8f"),
                   delimiter="    ",
                   header=header_str,
                   comments='')  # comments='' avoids prepending '#' to the header
        logging.info(f"Fermi energies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving Fermi energies: {e}")


def plot_fermi_energies(steps, fermi_energies):
    """Plots Fermi energies as a function of time steps."""
    plt.figure()
    plt.plot(steps, fermi_energies)
    plt.scatter(steps, fermi_energies, marker="o", s=80, edgecolors="black", zorder=2)
    plt.xlabel("Time (ps)")
    plt.ylabel("Fermi Energy (eV)")
    plt.show()


def compute_std_mean(fermi_energies):
    """
    Computes the standard deviation of the mean (standard error) of the Fermi energies.
    Returns 0.0 if fewer than two energies are provided.
    """
    if len(fermi_energies) < 2:
        return 0.0
    return np.std(fermi_energies, ddof=1) / np.sqrt(len(fermi_energies))


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract Fermi energies from CP2K output file."
    )
    parser.add_argument(
        "-f", "--frequency-collect-fermi", type=int, default=50,
        help="Frequency (time step interval) to collect Fermi energies (default: 50)"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="output.out",
        help="Input file name (default: output.out)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="fermi_energies.dat",
        help="Output file name (default: fermi_energies.dat)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot the extracted Fermi level as a function of time"
    )
    parser.add_argument(
        "-ts","--timestep", type=float, default=1.0,
        help="Timestep (in fs) used in the simulation. Used only for plot."
    )
    args = parser.parse_args()

    FREQUENCY_COLLECT_FERMI = args.frequency_collect_fermi
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output

    logging.info("Starting Fermi energy extraction.")

    fermi_energies = read_fermi_energies(INPUT_FILE, FREQUENCY_COLLECT_FERMI)
    if not fermi_energies:
        logging.warning("No Fermi energies collected. Exiting.")
        return

    steps = np.arange(
        FREQUENCY_COLLECT_FERMI,
        len(fermi_energies) * FREQUENCY_COLLECT_FERMI + 1,
        FREQUENCY_COLLECT_FERMI
    )

    save_fermi_energies(OUTPUT_FILE, steps, fermi_energies)

    std_mean = compute_std_mean(fermi_energies)
    logging.info(f"Standard deviation of the mean Fermi energy: {std_mean:.8f} eV")

    if args.plot:
        timestep = args.timestep
        steps = [x * timestep / 1000 for x in steps] # Convert in fs
        plot_fermi_energies(steps, fermi_energies)

    logging.info("Fermi energy extraction completed.")


if __name__ == "__main__":
    main()

