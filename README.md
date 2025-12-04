# Probabilistic Demand Forecasting in E-Commerce with Deep Learning Ensemble
**Author:** Alexander Mishin
**Institution:** Tilburg University
**Date:** December 2025

This repository contains the source code, analysis scripts, and supplementary materials for my Master's thesis, "Probabilistic Demand Forecasting in E-Commerce with Deep Learning Ensemble."

## Installation & Setup
**Prerequisite:** You must have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 1. Clone the Repository
```bash
git clone https://github.com/voron507/bma-in-retail-forecasting master_thesis
cd master_thesis
```

### 2. Create Conda Environments
```bash
conda env create -f environments/main_environment.yml
conda env create -f environments/deepar_tft_environment.yml
conda env create -f environments/deeptcn_environment.yml
```

### 3. Activate the Environment
```bash
conda activate master_env
```

## Usage & Replication
To run the full analysis pipeline, start here:

```bash
python src/main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

