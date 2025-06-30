# UCLA Capstone Project: Solar Device Metrics Analysis

## Project Overview
[TODO]

## Sample Dataset Information
- **Source**: Solar inverter metrics data
- **Size**: 3.3+ million records
- **Time Period**: 2025-06-25 07:00 to 2025-06-27 06:55
- **Content**: Device performance metrics including AC power, energy output, voltage, temperature, and other operational parameters

## Prerequisites

### Environment Setup
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required libraries: pandas, numpy, matplotlib, seaborn, datetime

### Dataset Setup
1. **Download the dataset** from one of the following sources:
   - [OneDrive Link](https://1drv.ms/u/s!Ag4DJ6hMoaoWx7BBOiGJucz0ZohHWQ?e=J1cJpK)
   - Attachment sent from Rafael on June 28, 2025
2. **Extract the dataset**:
   - Unzip the downloaded file
   - Place the extracted files in the `device_metrics/` directory
   - Ensure `device_metrics.csv` is located at `device_metrics/device_metrics.csv`

## Project Structure
```
capstone/
├── README.md                           # This file
├── device_metrics/                     # Dataset directory
│   └── device_metrics.csv             # Main dataset (after extraction)
├── device_metrics_analysis.ipynb      # Main analysis notebook
```