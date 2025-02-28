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

# Constants
AU_TO_EV = 27.2113838565563  # Hartree to eV conversion factor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def read_fermi_energies(filename, time_step_collect_fermi):
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
                if steps % time_step_collect_fermi == 0:
                    try:
                        fermi_energies.append(float(line.split()[-1]) * AU_TO_EV)
                    except ValueError as e:
                        logging.warning(f"Skipping line due to error: {line.strip()} - {e}")

    logging.info(f"Total simulation steps: {steps}")
    logging.info(f"Collected {len(fermi_energies)} Fermi energies")

    return fermi_energies, steps


def save_fermi_energies(filename, time_steps, fermi_energies):
    """Saves Fermi energies and corresponding time steps to a file with a header."""
    try:
        header_str = "TimeStep    FermiEnergy (eV)"
        np.savetxt(filename,
                   np.column_stack((time_steps, fermi_energies)),
                   fmt=("%.2f", "%.8f"),
                   delimiter="    ",
                   header=header_str,
                   comments='')  # comments='' avoids prepending '#' to the header
        logging.info(f"Fermi energies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving Fermi energies: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract Fermi energies from CP2K output file."
    )
    parser.add_argument(
        "-ts", "--time-step-collect-fermi", type=int, default=50,
        help="Time step interval to collect Fermi energies (default: 50)"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="output.out",
        help="Input file name (default: output.out)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="fermi_energies.dat",
        help="Output file name (default: fermi_energies.dat)"
    )
    args = parser.parse_args()

    TIME_STEP_COLLECT_FERMI = args.time_step_collect_fermi
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output

    logging.info("Starting Fermi energy extraction.")

    fermi_energies, steps = read_fermi_energies(INPUT_FILE, TIME_STEP_COLLECT_FERMI)
    if not fermi_energies:
        logging.warning("No Fermi energies collected. Exiting.")
        return

    time_steps = np.arange(
        TIME_STEP_COLLECT_FERMI,
        len(fermi_energies) * TIME_STEP_COLLECT_FERMI + 1,
        TIME_STEP_COLLECT_FERMI
    )

    save_fermi_energies(OUTPUT_FILE, time_steps, fermi_energies)

    logging.info("Fermi energy extraction completed.")


if __name__ == "__main__":
    main()

