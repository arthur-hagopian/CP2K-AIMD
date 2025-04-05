#!/usr/bin/env python3
# Author: Arthur Hagopian
# Contact: arth.hagopian@gmail.com
# Date: 04/2025

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONSTANTS
# -------------------------------
AU_TO_EV = 27.2113838565563  # Hartree to eV


# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def read_fermi_energies(filename, frequency):
    """
    Reads Fermi energies from a CP2K output file, keeping every `frequency`-th value.
    Returns a list of energies in eV.
    """
    print(f"Reading Fermi energies from file: {filename}")
    if not os.path.isfile(filename):
        print("Error: file not found.")
        return [], 0

    fermi_energies = []
    steps_seen = 0
    fermi_line_pattern = re.compile(r".*Fermi.*")

    with open(filename, "r") as f:
        for line in f:
            if fermi_line_pattern.match(line):
                steps_seen += 1
                if steps_seen % frequency == 0:
                    try:
                        energy = float(line.strip().split()[-1])
                        fermi_energies.append(energy * AU_TO_EV)
                    except ValueError:
                        print(f"Warning: Skipping malformed line: {line.strip()}")

    print(f"Finished reading file. Total matched steps: {steps_seen}, Fermi energies extracted: {len(fermi_energies)}")
    return fermi_energies, steps_seen


def save_fermi_energies(fermi_energies, frequency, timestep_fs, output_filename):
    """
    Save the extracted Fermi energies to a .dat file.
    Step column is in fs.
    """
    steps_fs = np.arange(frequency, frequency * len(fermi_energies) + 1, frequency) * timestep_fs
    print(f"Saving Fermi energy data to {output_filename} ...")
    header = "Time_fs    FermiEnergy_eV"
    np.savetxt(output_filename, np.column_stack((steps_fs, fermi_energies)), fmt='%10.2f %20.10f', header=header)
    return steps_fs


def plot_fermi_energies(time_fs, fermi_energies, sem):
    """
    Plot Fermi energy vs. time in ps and show standard error of the mean on the plot.
    """
    time_ps = np.array(time_fs) / 1000.0
    print("Plotting Fermi energy vs. time...")
    plt.figure(figsize=(8,6))
    plt.plot(time_ps, fermi_energies, marker='o', linestyle='-', color='blue')
    plt.xlabel("Time (ps)")
    plt.ylabel("Fermi Energy (eV)")
    plt.grid(True)
    plt.text(0.99, 0.98, f"SEM: {sem:.6f} eV", ha='right', va='top',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    plt.tight_layout()
    plt.show()


def compute_std_mean(values):
    """
    Compute the standard error of the mean.
    """
    if len(values) < 2:
        return 0.0
    return np.std(values, ddof=1) / np.sqrt(len(values))


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract Fermi energies from CP2K output.")
    parser.add_argument("-i", "--input", type=str, default="output.out", help="Input CP2K output file")
    parser.add_argument("-o", "--output", type=str, default="fermi_energies.dat", help="Output data file")
    parser.add_argument("-f", "--frequency", type=int, default=50, help="Sampling frequency (default: 50)")
    parser.add_argument("-ts", "--timestep", type=float, default=1.0, help="Timestep in fs (default: 1.0)")
    parser.add_argument("--plot", action="store_true", help="Plot the Fermi energy vs. time")
    args = parser.parse_args()

    # Step 1: Read energies
    fermi_energies, total_steps = read_fermi_energies(args.input, args.frequency)
    if not fermi_energies:
        print("No Fermi energies found. Exiting.")
        return

    # Step 2: Save to file
    time_fs = save_fermi_energies(fermi_energies, args.frequency, args.timestep, args.output)

    # Step 3: Compute std mean
    sem = compute_std_mean(fermi_energies)
    print(f"Standard error of mean Fermi energy: {sem:.8f} eV")

    # Step 4: Plot
    if args.plot:
        plot_fermi_energies(time_fs, fermi_energies, sem)

    print("All processing completed.")


if __name__ == '__main__':
    main()

