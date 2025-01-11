# Malware Detection Project

## Project Overview
This project aims to classify processes as either benign or malware based on various features.

## Project Structure
- `app.py`: Streamlit app for predicting malware or benign processes.
- `train_model.py`: Script to train the neural network model.
- `Malware dataset.csv`: Dataset used for training.
- `malware_model.h5`: Saved neural network model.
- `scaler.pkl`: Saved StandardScaler object for preprocessing.
- `requirements.txt`: Dependencies for running the project.
- `README.md`: Project documentation.

## How to Run

1. **Train the Model**
   Run the `train_model.py` to train the neural network and generate `malware_model.h5` and `scaler.pkl`.

   ```bash
   python train_model.py
