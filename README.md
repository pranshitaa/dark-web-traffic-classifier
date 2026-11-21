# 🛡️ Dark Web Traffic Classifier (DarkShield AI)

A production-ready cybersecurity machine learning system that classifies encrypted network traffic flows as either **Normal** or **Dark Web** (Tor / I2P proxy traffic).

---

## 📁 Project Architecture

```
dti/
├── data/
│   └── sample_traffic.csv          # Realistic network flow dataset
├── models/
│   ├── trained_model.pkl           # Saved Scikit-Learn / Random Forest model
│   ├── dl_model.pkl                # Saved Deep Neural Network (MLP) model
│   └── preprocessor.pkl            # Scaler and preprocessing state
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Feature scaling, imputation, and encoding
│   ├── train.py                    # Modular CLI model training script
│   └── predict.py                  # End-to-end inference & confidence scoring
├── tests/
│   ├── __init__.py
│   └── test_system.py              # Automated pytest suite
├── .streamlit/
│   └── config.toml                 # Cybersecurity dark theme configuration
├── app.py                          # Streamlit threat intelligence dashboard
├── requirements.txt                # Project dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Activate Environment & Install Dependencies
```bash
# In your terminal
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
Train the Random Forest classifier on network flow signatures:
```bash
python src/train.py --model_type sklearn --data data/sample_traffic.csv
```

Or train the Deep Neural Network model:
```bash
python src/train.py --model_type dl --data data/sample_traffic.csv
```

### 3. Launch the Cybersecurity Dashboard
```bash
streamlit run app.py
```
This will open the dashboard in your web browser. You can upload any network traffic CSV or click **"Load Built-in Sample Traffic"** in the sidebar to run analysis immediately!

---

## 🧪 Run Automated Tests
Run the complete test suite to verify data validation, error handling, preprocessing, and model inference:
```bash
pytest tests/ -v
```

---

## 📊 Features & Threat Capabilities

- **Deep Flow Inspection**: Analyzes `flow_duration`, `packet_size`, `inter_arrival_time`, and `byte_count`.
- **Confidence Scoring**: Each flow classification includes a calibrated probability score.
- **Cyber Threat Visualizations**:
  - Donut distribution of threat vs benign traffic.
  - Model confidence distribution histogram.
  - Interactive behavioral flow scatter plot (Duration vs Packet Size).
- **Automated Alerts**: Real-time alert banners highlighting detected Tor/Dark Web anonymization signatures.
- **Forensic Export**: One-click download of the complete audited classification report in CSV format.
