# Predictive Maintenance for Solar Plant Inverters

## 📌 Overview
This project is part of the UCLA MEng Capstone Program, in collaboration with **MN8 Energy (a Goldman Sachs-backed renewable energy company)**.  
We are building a **machine learning–driven predictive maintenance system** for solar plant inverters, designed to detect potential failures several days before they occur.

Key features:
- End-to-end ML pipeline on **Microsoft Azure** for data ingestion, preprocessing, model training, and deployment
- Time-series analysis on **3M+ sensor records** across multiple inverters
- Deep learning models (CNN-LSTM) for failure prediction
- Early warning system that can issue alarms **5 days before failure events** with ~77% recall (ongoing improvement)
- System under deployment in a **California solar plant** for field evaluation and dynamic tuning

---

## 📂 Repository Structure
```
predictive_maintenance/
│── azure/                        # Azure ML Studio integration scripts
│── dataset/                      # Dataset preparation and evaluation notebooks
│── model/                        # Model definitions and saved models
│── plot/                         # Plotting and visualization utilities
│── inverter_predictive_maintenance/  # Core package (pipeline, utilities)
│── .gitignore                    # Git ignore rules
│── README.md                     # Project introduction (this file)
│── evaluate_model.ipynb          # Notebook for evaluating trained models
│── failure_sessions_w_maintenance.csv  # Failure sessions dataset with maintenance info
│── label_maintainance_session.ipynb    # Notebook for labeling maintenance sessions
│── train_model_local.ipynb       # Local training notebook
```

---

## ⚙️ Tech Stack
- **Languages:** Python (NumPy, Pandas, PyTorch, Scikit-learn), SQL  
- **Cloud & Infrastructure:** Microsoft Azure (Data Factory, ML Studio, SQL Database, Blob Storage)  
- **Machine Learning:** CNN-LSTM
- **Visualization:** Plotly, Matplotlib, Mermaid / Draw.io for diagrams  
- **Documentation:** LaTeX, Markdown, GitHub

---

## 🚀 Setup & Usage
### 1. Clone the repository
```bash
git clone https://github.com/libao3128/predictive_maintenance.git
cd predictive_maintenance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
or using Conda:
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
Run `label_maintainance_session.ipynb`. This will generate two CSV files in the `dataset/` folder:
```
dataset/failure_sessions.csv              # Raw labeled sessions (unfiltered)
dataset/failure_sessions_w_maintenance.csv # Filtered failure sessions
```

### 5. Data Preprocessing
Run `data_pipeline.ipynb` to preprocess the raw inverter data. Ensure the following path exists:
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
Run `train_model_local.ipynb` to execute the ML pipeline and train the predictive model.  
The trained model checkpoints will be saved under the `model/` folder.

### 7. Evaluate Results
Run `evaluate_model.ipynb` to evaluate the trained model and review performance metrics.


---

## 👥 Team
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

## TODO
- [] Refine InverterTimeSeriesDataset to support metadata record. (Angela's features)
- [] Add temporal smoothing post process. (Grace's features)