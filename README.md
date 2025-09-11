# Predictive Maintenance for Solar Plant Inverters

## Overview
This project is part of the UCLA MEng Capstone Program, in collaboration with **MN8 Energy (a Goldman Sachs-backed renewable energy company)**.  
We are building a **machine learning–driven predictive maintenance system** for solar plant inverters, designed to detect potential failures several days before they occur.

Key features:
- End-to-end ML pipeline on **Microsoft Azure** for data ingestion, preprocessing, model training, and deployment
- Time-series analysis on **3M+ sensor records** across multiple inverters
- Deep learning models (CNN-LSTM) for failure prediction
- Early warning system that can issue alarms **5 days before failure events** with ~77% recall (ongoing improvement)
- System under deployment in a **California solar plant** for field evaluation and dynamic tuning

---

## Repository Structure
```
predictive_maintenance/
├── azure/                        # Azure ML Studio integration scripts
│   ├── data_factory/             # Azure Data Factory configurations
│   └── ml_studio/                # ML Studio notebooks and scripts
├── config/                       # Configuration files
│   ├── dataset_parameters.json   # Dataset configuration
│   └── model_parameters.json     # Model configuration
├── dataset/                      # Dataset files and utilities
│   ├── inverter_data/           # Raw inverter sensor data (parquet files)
│   ├── failure_sessions.csv     # Labeled failure sessions
│   ├── failure_sessions_w_maintenance.csv  # Filtered failure sessions
│   ├── train_data.csv           # Training dataset
│   ├── val_data.csv             # Validation dataset
│   ├── test_data.csv            # Test dataset
│   └── README.md                # Dataset documentation
├── inverter_predictive_maintenance/  # Core Python package
│   ├── dataset/                 # Dataset handling modules
│   ├── model/                   # Model architectures and losses
│   ├── preprocess/              # Data preprocessing modules
│   ├── training/                # Training utilities
│   ├── metrics/                 # Evaluation metrics
│   ├── visualize/               # Visualization tools
│   ├── postprocess/             # Post-processing utilities
│   └── cli.py                   # Command line interface
├── model/                       # Saved model checkpoints
├── plot/                        # Generated visualization plots
├── scripts/                     # Jupyter notebooks for data pipeline
│   ├── data_pipeline.ipynb     # Data preprocessing pipeline
│   ├── label_failure_sessions.ipynb  # Failure session labeling
│   └── train_model.ipynb       # Model training notebook
├── evaluate_model.ipynb         # Model evaluation notebook
├── setup.py                     # Package setup script
├── pyproject.toml              # Modern Python project configuration
├── requirements.txt             # Python dependencies
├── environment.yml             # Conda environment file
└── README.md                   # Project documentation (this file)
```

---

## Tech Stack

### Core Technologies
- **Languages:** Python 3.8+ (NumPy 2.3.2, Pandas 2.3.1, PyTorch 2.8.0, Scikit-learn 1.7.1), SQL  
- **Cloud & Infrastructure:** Microsoft Azure (Data Factory, ML Studio, SQL Database, Blob Storage)  
- **Machine Learning:** CNN-LSTM hybrid architecture, Focal Loss, Weighted BCE Loss
- **Data Processing:** PyArrow 21.0.0 (Parquet support), SciPy 1.16.1
- **Visualization:** Plotly 6.2.0, Matplotlib 3.10.6
- **Development:** Jupyter Notebooks, IPython 9.4.0, IPykernel 6.29.5

---

## Setup & Usage
### 1. Clone the repository
```bash
git clone https://github.com/libao3128/predictive_maintenance.git
cd predictive_maintenance
```

### 2. Install dependencies

#### Option A: Install as a Python package (Recommended)
```bash
# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,azure]"  # Include development and Azure tools
```

#### Option B: Install dependencies only
```bash
pip install -r requirements.txt
```

#### Option C: Using Conda
```bash
conda env create -f environment.yml
conda activate predictive_maintenance
```

### 3. Prerequisites
The **raw dataset is not public**. Please contact our team members or MN8 Energy to request access.  
Refer to the main page `README.md` for contact details.

Expected files in this directory:
```
dataset/inverter_data/*.parquet   # Processed inverter data from Microsoft Azure Data Factory
```

### 4. Label Failure Sessions
Run `scripts/label_failure_sessions.ipynb`. This will generate two CSV files in the `dataset/` folder:
```
dataset/failure_sessions.csv              # Raw labeled sessions (unfiltered)
dataset/failure_sessions_w_maintenance.csv # Filtered failure sessions
```

### 5. Data Preprocessing
Run `scripts/data_pipeline.ipynb` to preprocess the raw inverter data. Ensure the following path exists:
```
dataset/inverter_data/*.parquet
```
After preprocessing, you will obtain:
```
dataset/train_data.csv
dataset/val_data.csv
dataset/test_data.csv
```

### 6. Train Model
Run `scripts/train_model.ipynb` to execute the ML pipeline and train the predictive model.  
The trained model checkpoints will be saved under the `model/` folder.

### 7. Evaluate Results
Run `evaluate_model.ipynb` to evaluate the trained model and review performance metrics.

### 8. Command Line Interface (CLI)
The package includes a command-line interface for common operations:

```bash
# Visualize inverter data and failure sessions
inverter-predictive-maintenance visualize \
    --data-path dataset/inverter_data \
    --failures dataset/failure_sessions.csv \
    --output plot/

# Prepare dataset for training
inverter-predictive-maintenance prepare \
    --data-path dataset/inverter_data \
    --failures dataset/failure_sessions.csv \
    --output dataset/processed/ \
    --pre-days 5

# Train model from command line
inverter-predictive-maintenance train \
    --train-data dataset/train_data.csv \
    --val-data dataset/val_data.csv \
    --output model/ \
    --epochs 50

# Evaluate trained model
inverter-predictive-maintenance evaluate \
    --model-path model/best_model.pth \
    --test-data dataset/test_data.csv
```

For more CLI options and help:
```bash
inverter-predictive-maintenance --help
inverter-predictive-maintenance <command> --help
```

---

## Team
- **Li-Chun Huang**, leo900527@gmail.com
- **Grace Cheng**
- **Portia Huang**
- **Yen-Yun Kuo**

Advisors:
- **Ramon Millan**, MN8 Energy  
- **Camilo Lombo**, MN8 Energy  
- **Bruce Huang**, UCLA Samueli School of Engineering
- **Joey Lao**, UCLA Samueli School of Engineering

---

## Project Status & TODO

### Completed Features
- [x] **Core Package Structure**: Complete Python package with modular design
- [x] **Dataset Management**: InverterTimeSeriesDataset with metadata support
- [x] **Model Architecture**: CNN-LSTM hybrid model with advanced loss functions
- [x] **Data Preprocessing**: Comprehensive data loading, cleaning, and labeling pipeline
- [x] **Training Pipeline**: Complete training loop with validation and checkpointing
- [x] **Evaluation Metrics**: Top-K accuracy, AUCPR, and time series specific metrics
- [x] **Visualization Tools**: Interactive plots and failure session visualization
- [x] **Command Line Interface**: CLI for common operations
- [x] **Azure Integration**: Data Factory and ML Studio integration
- [x] **Package Distribution**: setup.py and pyproject.toml configuration

### In Progress
- [ ] **Temporal Smoothing**: Post-processing for consecutive run filtering
- [ ] **Documentation**: API documentation and user guides

### Future Enhancements
- [ ] **Real-time Monitoring**: Live data streaming and real-time predictions
- [ ] **Multi-site Support**: Scaling to multiple solar plant locations
- [ ] **Advanced Models**: Transformer-based architectures for time series
- [ ] **Automated Retraining**: Continuous learning and model updates
- [ ] **Dashboard Interface**: Web-based monitoring and alert system

---

## License & Links

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this project in your research, please cite:
```bibtex
@software{predictive_maintenance_2024,
  title={Predictive Maintenance for Solar Plant Inverters},
  author={UCLA MEng Capstone Team},
  year={2024},
  url={https://github.com/libao3128/predictive_maintenance},
  note={UCLA MEng Capstone Program in collaboration with MN8 Energy}
}
```

### Acknowledgments
- **MN8 Energy** (Goldman Sachs-backed renewable energy company) for providing the dataset and domain expertise
- **UCLA Samueli School of Engineering** for academic support and guidance
- **Microsoft Azure** for cloud infrastructure and ML services