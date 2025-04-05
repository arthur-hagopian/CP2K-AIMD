# Arthur Hagopian <arthur.hagopian@umontpellier.fr>, version 06/07/2023

# INPUT FILE NEEDED : "trajectories.xyz"
# Should contain trajectories of the whole simulation (concatenated file)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_parameters
import MDAnalysis as mda
from pathlib import Path
from scipy.constants import Avogadro
import sys
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# DEFINITIONS
type_surface_atoms = 'Pt'
num_surface_atoms = 20
x_min = 0
y_min = 0
z_min = 0
x_max = 13.8593
y_max = 9.602
z_max = 54.2896
start_analysis  = int(5000/5)
stop_analysis = int(15000/5)
num_bins = 300

# Define universe
u = mda.Universe("trajectories.xyz")
# Define electrode surface atoms
electrode_surface_atoms = u.select_atoms(f'name {type_surface_atoms}')[-num_surface_atoms:]
# Deifne O atoms
o_atoms = u.select_atoms('name O')

# Initialize variables
num_timesteps = len(u.trajectory[start_analysis:stop_analysis])
o_atoms_counts = np.zeros(num_bins)

# Iterate over each timestep : define electrode surface position
electrode_surface_atoms_avg_per_frame = []
for ts in u.trajectory[start_analysis:stop_analysis]:

    electrode_surface_atoms_positions = electrode_surface_atoms.positions[:, 2] # Z coordinates
    electrode_surface_atoms_avg_per_frame.append(sum(electrode_surface_atoms_positions) / len(electrode_surface_atoms_positions))

electrode_surface_atoms_avg = sum(electrode_surface_atoms_avg_per_frame) / len(electrode_surface_atoms_avg_per_frame)
print(f'Electrode surface position : {electrode_surface_atoms_avg}')

z_max = z_max - electrode_surface_atoms_avg
bin_edges = np.linspace(z_min, z_max, num_bins + 1)

# Iterate over each timestep : collect O atoms positions
o_atoms_counts = np.zeros(num_bins)
for ts in u.trajectory[start_analysis:stop_analysis]:

    o_atoms_positions = o_atoms.positions[:, 2]
    o_atoms_positions = [z - electrode_surface_atoms_avg for z in o_atoms_positions]

    # Bin the O atoms based on their z-positions
    o_atoms_bin_indices = np.searchsorted(bin_edges, o_atoms_positions) # Associate positions to bin indices
    o_atoms_counts += np.bincount(o_atoms_bin_indices, minlength=num_bins) # Add 1 every time a bin is occupied

    non_zero_idx=np.nonzero(o_atoms_counts)
    non_zero_elements=o_atoms_counts[non_zero_idx]

# Calculate the average count per bin
#print(o_atoms_counts)
avg_o_atoms_counts = o_atoms_counts / num_timesteps

print(avg_o_atoms_counts)

# Calculate the concentration per bin
num_h2o_per_ang3 = 55.5 * 1e-27 * Avogadro # Molarity of water = 55.5 mol.L-1 && 1 ang^3 = 1e-27 L
num_particles_per_mol_per_ang3 = Avogadro * 1e-27
volume_bin = (bin_edges[1] - bin_edges[0]) * (x_max - x_min) * (y_max - y_min) # In ang^3
num_h2o_per_bin = volume_bin * num_h2o_per_ang3

o_atoms_concentration = (avg_o_atoms_counts / volume_bin) / num_particles_per_mol_per_ang3 * 18.01528 / 1000

# Define x-axis
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

### PRINT IN water_density.dat ###
file_out = open("water_density.dat", "w")
i = 0
while i < len(o_atoms_concentration):
    file_out.write("%12.10f" % (bin_centers[i]) + "    " + "%12.10f" % (o_atoms_concentration[i]) + "\n")
    i += 1
file_out.close()
print('... Water density data printed in file : fermi_energies.dat\n')

# Plot the average count per bin
fig, ax =plt.subplots()
#ax.axvline(x = 0, color = 'grey', linewidth=3, label = 'Electrode surface')
#ax.bar(bin_centers, o_atoms_concentration, width=(bin_edges[1] - bin_edges[0]), alpha=0.8, label='O atoms', color=matplotlib_parameters.default[1])
ax.plot(bin_centers, o_atoms_concentration, color="#aeb6bf")
ax.fill_between(bin_centers, o_atoms_concentration, 0, color="#aeb6bf", alpha=0.8)
ax.set_xlabel('Distance from electrode surface ($\mathrm{\AA}$)')
ax.set_ylabel(r'$\mathrm{\rho_{O}}$ ($\mathrm{g.cm^{-3}}$)')

#Calculate and print bulk concentration
print(f"O atoms concentration: {o_atoms_concentration[(bin_centers >=0) & (bin_centers<=25)].mean()}")

#Define max x and max y
#ax.set_ylim([0,0.10])
ax.set_xlim([0,26])
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(which='major', length=8, width=2)
ax.tick_params(which='minor', length=4, width=1)

ax.legend()
plt.show()

