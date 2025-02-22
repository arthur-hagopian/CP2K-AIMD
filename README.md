# CP2K-AIMD

## Overview
This repository contains a package for analyzing **Ab Initio Molecular Dynamics (AIMD) simulations** performed with **CP2K**. The tools provided here facilitate data extraction, processing, and visualization of key simulation outputs. The tools are appropriate for the simulations of **metal/electrolyte interfaces**.

## Repository Structure
- **scripts**: Python scripts for data extraction, processing and visualization
- **examples**: Example data and usage scripts
- **README.md**: Project documentation
- **requirements.txt**: Required Python packages

## Installation
To use the scripts, clone the repository and install dependencies:
```sh
git clone https://github.com/arthur-hagopian/AIMD-CP2K.git
cd AIMD-CP2K
pip install -r requirements.txt
```

## Usage
Run individual scripts to analyze your CP2K output files. For example, to extract the Fermi energy:
```sh
python scripts/extract_fermi_energy.py path/to/your/cp2k_output.log
```

## Contributing
Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.

## Contact
For any questions or suggestions, feel free to reach out via GitHub issues or email.

---
🚀 **Happy simulating!**

