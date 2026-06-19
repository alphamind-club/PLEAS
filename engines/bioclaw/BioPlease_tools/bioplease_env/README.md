# BioPLEASE Environment Setup

This directory contains scripts and configuration files to set up a comprehensive bioinformatics environment with various tools and packages.

1. Clone the repository:
   ```bash
   git clone https://github.com/snap-stanford/BioPLEASE.git
   cd BioPLEASE/bioplease_env
   ```

2. Setting up the environment:
- (a) If you want to use or try out the basic agent without the full E1 or install your own softwares, run the following script:

```bash
conda env create -f environment.yml
```

- (b) If you want to use the full environment E1, run the setup script (this script takes > 10 hours to setup, and requires a disk of at least 30 GB quota). Follow the prompts to install the desired components.

```bash
bash setup.sh
```

If you already installed the base version, and just wants to add the additional packages in the new release, you can simply do:

```bash
bash new_software_v005.sh
```

Note: we have only tested this setup.sh script with Ubuntu 22.04, 64 bit.


3. Lastly, to activate the bioplease environment:
```bash
conda activate bioplease_e1
```
