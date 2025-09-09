## Expected Files in Directory
Before you run the following script, make sure you have request the raw data and put them in 
   ```
   dataset\inverter_data/*.parquet # Processed inverter data via Microsoft Azure Data Factory
   ```

### Labeling Failure Sessions
Upon completion of `label_failure_sessions.ipynb`, the following files will be generated:
   ```
   dataset\failure_sessions.csv
   dataset\failure_sessions_w_maintenance.csv
   ```

### Data Preprocessing
After executing the preprocessing script `data_pipeline.ipynb`, the following files will be created:
   ```
   dataset\train_data.csv
   dataset\val_data.csv
   dataset\test_data.csv
   ```
