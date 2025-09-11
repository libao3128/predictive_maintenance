# Inverter Predictive Maintenance Package

A comprehensive Python package for predictive maintenance of solar plant inverters, providing tools for data preprocessing, model training, evaluation, and visualization.

This package is developed as part of the UCLA MEng Capstone Program in collaboration with MN8 Energy (a Goldman Sachs-backed renewable energy company).

## Package Structure

```
inverter_predictive_maintenance/
├── __init__.py                 # Main package initialization
├── dataset/                    # Dataset handling modules
│   ├── __init__.py
│   ├── base.py                # Base dataset classes
│   ├── specialized.py         # Specialized dataset classes
│   └── utils.py               # Dataset utility functions
├── model/                     # Model architectures and losses
│   ├── __init__.py
│   ├── networks.py            # Neural network architectures
│   └── losses.py              # Loss functions
├── preprocess/                # Data preprocessing modules
│   ├── __init__.py
│   ├── data_loading.py        # Data loading utilities
│   ├── labeling.py            # Data labeling functions
│   ├── cleaning.py            # Data cleaning utilities
│   └── splitting.py           # Data splitting functions
├── training/                  # Training utilities
│   ├── __init__.py
│   ├── trainer.py             # Training loops
│   └── evaluation.py          # Evaluation functions
├── metrics/                   # Evaluation metrics
│   ├── __init__.py
│   ├── classification.py     # Classification metrics
│   └── time_series.py         # Time series specific metrics
├── visualize/                 # Visualization tools
│   ├── __init__.py
│   ├── time_series.py         # Time series visualization
│   └── training.py            # Training visualization
└── postprocess/               # Post-processing utilities
    ├── __init__.py
    └── smoothing.py           # Prediction smoothing
```

## Key Features

### 1. Dataset Management
- **Base Dataset Classes**: `InverterTimeSeriesDataset` and `InverterTimeSeriesDataset_metadata`
- **Specialized Datasets**: Positive-only and negative-only filtering
- **Windowing**: Automatic sliding window creation for time series
- **Metadata Tracking**: Comprehensive metadata for each data sample

### 2. Model Architectures
- **CNNLSTMModel**: CNN-LSTM hybrid architecture for time series classification
- **FocalLoss**: Advanced loss function for handling class imbalance
- **WeightedBCELoss**: Weighted binary cross-entropy loss

### 3. Data Preprocessing
- **Data Loading**: Support for parquet files and CSV data
- **Labeling**: Automatic pre-failure period labeling
- **Cleaning**: Missing value imputation and data cleaning
- **Splitting**: Temporal train-test splitting

### 4. Training & Evaluation
- **Training Loops**: Comprehensive training with validation
- **Metrics**: Top-K accuracy, AUCPR, and time series specific metrics
- **Evaluation**: Detailed model performance assessment

### 5. Visualization
- **Time Series Plots**: Interactive plots with failure session overlays
- **Training Curves**: Loss and metric visualization
- **Timeline Views**: Failure session timeline visualization

### 6. Post-Processing
- **Smoothing**: Consecutive run filtering and noise reduction
- **Event Detection**: Advanced event detection algorithms

## Usage Examples

### Basic Dataset Creation
```python
import inverter_predictive_maintenance as ipm
from inverter_predictive_maintenance import InverterTimeSeriesDataset

# Create dataset from DataFrame
dataset = InverterTimeSeriesDataset.from_dataframe(
    dataframe=df,
    feature_cols=['voltage', 'current', 'power'],
    label_col='label',
    window_size=30,
    stride=1
)
```

### Model Training
```python
from inverter_predictive_maintenance import CNNLSTMModel, FocalLoss, train_loop

# Initialize model and loss
model = CNNLSTMModel(num_features=10)
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# Train model
log = train_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=50
)
```

### Data Preprocessing
```python
from inverter_predictive_maintenance.preprocess import load_parquet_data, prepare_dataset

# Load data
inverter_data = load_parquet_data('data/parquet_files/')
failure_sessions = load_failure_sessions('data/failures.csv')

# Prepare labeled dataset
labeled_data = prepare_dataset(
    inverter_data=inverter_data,
    failure_sessions=failure_sessions,
    pre_days=5
)
```

### Visualization
```python
from inverter_predictive_maintenance.visualize import visualize_mean_values, visualize_failure_timeline

# Create time series visualizations
visualize_mean_values(
    inverter_data=inverter_data,
    failure_sessions=failure_sessions,
    feature_cols=['voltage', 'current'],
    freq='H'
)

# Create failure timeline
visualize_failure_timeline(failure_sessions)
```

## Design Principles

### 1. Single Responsibility Principle
Each module has a clear, focused responsibility:
- `dataset/`: Data handling and windowing
- `model/`: Neural network architectures
- `preprocess/`: Data preprocessing
- `training/`: Training and evaluation
- `metrics/`: Performance evaluation
- `visualize/`: Visualization tools
- `postprocess/`: Post-processing utilities

### 2. Comprehensive Type Hints
All functions include complete type annotations for better code clarity and IDE support.

### 3. Extensive Documentation
Every function includes detailed docstrings with:
- Purpose and functionality
- Parameter descriptions
- Return value descriptions
- Usage examples
- Error conditions

### 4. Error Handling
Robust error handling with informative error messages for common issues.

### 5. Modularity
Clean separation of concerns allows for easy testing, maintenance, and extension.

## Installation

### From Source
```bash
git clone https://github.com/libao3128/predictive_maintenance.git
cd predictive_maintenance
pip install -e .
```

### Direct Import
```python
import sys
sys.path.append('path/to/predictive_maintenance')
import inverter_predictive_maintenance as ipm
from inverter_predictive_maintenance import InverterTimeSeriesDataset, CNNLSTMModel
```

### Command Line Interface
```bash
# Visualize data
inverter-predictive-maintenance visualize --data-path dataset/inverter_data --failures dataset/failures.csv

# Prepare dataset
inverter-predictive-maintenance prepare --data-path dataset/inverter_data --failures dataset/failures.csv --output dataset/processed/
```

## Dependencies

- `torch`: PyTorch for deep learning
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities
- `plotly`: Interactive visualizations
- `matplotlib`: Static visualizations
- `tqdm`: Progress bars
- `scipy`: Scientific computing (for post-processing)

## Contributing

When adding new functionality:
1. Follow the existing module structure
2. Add comprehensive type hints
3. Include detailed docstrings
4. Add appropriate error handling
5. Update the relevant `__init__.py` files
6. Test thoroughly before committing

## License

This package is part of the predictive maintenance project for inverter systems.
