# VPN Network Traffic Sampling Benchmark

This project provides a comprehensive framework for evaluating various Machine Learning (ML) and Deep Learning (DL) sampling techniques on network traffic data. It focuses on addressing class imbalance in VPN traffic datasets using samplers like SMOTE, WGAN-GP, CTGAN, and more.

## Project Structure

```text
final_project/
├── data/                       # Dataset files
│   └── NetTrafficData.csv
├── docs/                       # Project documentation and research papers
│   ├── split_configs/          # Details of various data split ratios
├── notebooks/                  # Experimentation notebooks
│   ├── 60-40/                  # Experiments with 60% train / 40% test split
│   ├── 70-30/                  # Experiments with 70% train / 30% test split
│   ├── 80-20/                  # Experiments with 80% train / 20% test split
│   └── MISC.ipynb              # Miscellaneous experiments
├── traffic_utils/              # Core utility package for the project
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── ml_samplers.py          # Traditional ML samplers (SMOTE, etc.)
│   ├── dl_samplers.py          # Deep Learning samplers (WGAN, etc.)
│   ├── models.py               # Model training and evaluation logic
│   └── visualizations.py       # Plotting utilities
├── results/                    # Generated plots and performance comparisons
├── requirements.txt            # List of Python dependencies
└── .gitignore                  # Git ignore rules
```

## Getting Started

### Prerequisites
- Python 3.8+
- Recommended: A virtual environment (venv or conda)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Sriram-ai-prog/VPN-Traffic-Sampling-Benchmark
   cd final_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments
The project is organized into three main split scenarios (60-40, 70-30, and 80-20). Each scenario has a set of notebooks implementing different sampling techniques.

Navigate to `notebooks/` and open any notebook in Jupyter Lab or VS Code to run the experiments.

## Core Package: `traffic_utils`
The `traffic_utils` directory is a local Python package that contains all the modularized logic. The notebooks automatically add the project root to `sys.path` to import these utilities.

## Dataset
The project uses `NetTrafficData.csv`, which should be placed in the `data/` directory.

## Contributors
- **Tadishetty Sriram**
- **Akhiranad Boddani**
- **Thota Ritvika reddy**

**Affiliation:** Bhavan's Vivekananda College of Science, Humanities and Commerce

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
