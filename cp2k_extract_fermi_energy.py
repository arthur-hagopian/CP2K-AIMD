#!/sw/arch/RHEL8/EB_production/2023/software/Python/3.11.3-GCCcore-12.3.0/bin/python
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 02/2025
# Description: Script to extract and analyze Fermi energies from CP2K output.
# Requirements: NumPy, Matplotlib

import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# Constants
AU_TO_EV = 27.2113838565563  # Hartree to eV conversion factor
TIME_STEP_COLLECT_FERMI = 50
INPUT_FILE = "output.out"
OUTPUT_FILE = "fermi_energies.dat"
PLOT_XLABEL = "Time (ps)"
PLOT_YLABEL = "Fermi energy (eV)"


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
    """Saves Fermi energies and corresponding time steps to a file."""
    try:
        np.savetxt(filename, np.column_stack((time_steps, fermi_energies)), fmt="%12.10f", delimiter="    ")
        logging.info(f"Fermi energies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving Fermi energies: {e}")


def plot_fermi_energies(time_steps, fermi_energies):
    """Plots Fermi energies against time steps."""
    plt.figure(figsize=(8, 6))
    plt.plot(time_steps / 1000, fermi_energies, color="blue", linewidth=1, linestyle="-", alpha=0.5, label="Fermi Energy")
    plt.scatter(time_steps / 1000, fermi_energies, marker="o", s=80, edgecolors="black", color="blue", alpha=1)
    plt.xlabel(PLOT_XLABEL)
    plt.ylabel(PLOT_YLABEL)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def main():
    logging.info("Starting Fermi energy extraction.")

    fermi_energies, steps = read_fermi_energies(INPUT_FILE, TIME_STEP_COLLECT_FERMI)
    if not fermi_energies:
        logging.warning("No Fermi energies collected. Exiting.")
        return

    time_steps = np.arange(TIME_STEP_COLLECT_FERMI, len(fermi_energies) * TIME_STEP_COLLECT_FERMI + 1, TIME_STEP_COLLECT_FERMI)

    save_fermi_energies(OUTPUT_FILE, time_steps, fermi_energies)
    plot_fermi_energies(time_steps, fermi_energies)

    logging.info("Fermi energy extraction completed.")


if __name__ == "__main__":
    main()

