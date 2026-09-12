# 🚚 Porter Delivery ETA Prediction using Neural Networks

## 📌 Overview

Porter needs to provide customers with reliable estimates of how long an order will take to reach its destination.

This project uses a **Neural Network regression model** to predict delivery time based on order characteristics, store information, delivery-partner availability, outstanding orders, driving duration, and time-based features.

The project demonstrates an **end-to-end machine learning solution**, from preprocessing and model training to production inference, FastAPI API development, validation, and Streamlit deployment.

---

## 🎯 Objective

* Predict **estimated delivery time in minutes**
* Use order and operational information to improve delivery-time estimation
* Build a **production-ready inference pipeline**
* Expose the model through a **FastAPI REST API**
* Provide an interactive **Streamlit application**
* Validate prediction consistency between API and Streamlit implementations

---

## 🌐 Live Application

🚀 **Streamlit App:** *https://porter-neural-network-pwer22gy2ztoxsm8gz4c5s.streamlit.app/*

Users can enter order and operational information and get:

* Predicted delivery time
* Model information
* Real-time prediction from the trained Neural Network

---

## 📊 Dataset Description

The dataset contains order-level and operational features related to food delivery.

Key features include:

* **Market ID** – Identifier for the market/location
* **Store Category** – Primary category of the store
* **Order Protocol** – Ordering protocol used for the order
* **Total Items** – Number of items in the order
* **Subtotal** – Total order value
* **Distinct Items** – Number of unique items
* **Item Prices** – Minimum and maximum item prices
* **On-Shift Dashers** – Number of delivery partners currently on shift
* **Busy Dashers** – Number of delivery partners currently busy
* **Outstanding Orders** – Number of pending orders
* **Driving Duration** – Estimated store-to-consumer driving duration
* **Created At** – Order creation timestamp

The target variable is:

**Estimated Delivery Time = Actual Delivery Time − Order Creation Time**

measured in **minutes**.

---

## 🧠 Approach

### 1. Data Preprocessing

* Handled missing and inconsistent values
* Converted timestamp information into usable features
* Applied categorical encoding
* Applied feature scaling using **StandardScaler**
* Preserved the exact preprocessing artifacts for production inference

---

### 2. Feature Engineering

Extracted time-based features from the order creation timestamp:

* **Hour of Day**
* **Day of Week**

Categorical features were transformed using **One-Hot Encoding**.

The final production model uses **75 input features**.

---

### 3. Neural Network

A feed-forward Neural Network was developed using **TensorFlow/Keras**.

Architecture:

```text
Input Layer
    ↓
Dense Layer — 64 neurons
    ↓
Dense Layer — 32 neurons
    ↓
Output Layer — 1 neuron
    ↓
Predicted Delivery Time
```

The model performs a **regression task**, predicting delivery time in minutes.

---

### 4. Production Inference

The trained model and preprocessing objects are serialized as artifacts:

```text
artifacts/
├── model.keras
├── scaler.pkl
├── feature_columns.pkl
└── model_metadata.json
```

The inference pipeline ensures that the same:

* Feature engineering
* Encoding
* Feature ordering
* Scaling
* Model

are used during production prediction.

---

## 🔌 FastAPI REST API

The trained model is also exposed through a **FastAPI REST API**.

### API Endpoint

```text
POST /predict
```

Example request:

```json
{
  "market_id": 1,
  "created_at": "2018-02-02T15:00:00",
  "store_primary_category": 1,
  "order_protocol": 1,
  "total_items": 2,
  "subtotal": 500,
  "num_distinct_items": 2,
  "min_item_price": 100,
  "max_item_price": 300,
  "total_onshift_dashers": 10,
  "total_busy_dashers": 5,
  "total_outstanding_orders": 5,
  "estimated_store_to_consumer_driving_duration": 1000
}
```

Example response:

```json
{
  "predicted_delivery_time_minutes": 51.32
}
```

### API Documentation

FastAPI automatically provides interactive Swagger documentation at:

```text
http://127.0.0.1:8001/docs
```

---

## 🔄 Prediction Validation

The project implements two independent inference paths:

```text
                 Same Input
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
     Streamlit              FastAPI
     Inference              Inference
          │                     │
          └──────────┬──────────┘
                     ↓
              Compare Results
```

The same input was passed through both implementations.

The predictions produced by **Streamlit and FastAPI were identical**, validating consistency across:

* Feature engineering
* One-hot encoding
* Feature alignment
* Scaling
* Neural Network inference

This helps ensure that the deployed Streamlit application reproduces the behavior of the FastAPI inference pipeline.

---

## 🚀 Deployment Architecture

```text
User → Streamlit Cloud → Preprocessing → Neural Network → Delivery ETA
```

The FastAPI implementation is maintained separately to demonstrate production-style model serving:

```text
Client → FastAPI → Validation → Preprocessing → Neural Network → Prediction
```

The Streamlit deployment performs inference directly using the serialized model and preprocessing artifacts.

---

## 🧩 Model & Artifact Validation

The application validates that the model artifacts remain compatible before generating predictions.

Validation includes:

* **75 expected feature columns**
* Scaler feature count
* Neural Network input dimensions
* Availability of required model artifacts

This prevents incompatible model, scaler, and feature configurations from silently producing incorrect predictions.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **FastAPI**
* **Pydantic**
* **Uvicorn**
* **Streamlit**
* **Jupyter Notebook**
* **Git / GitHub**

---

## 📂 Project Structure

```text
Porter Neural Networks/
│
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── predictor.py        # Preprocessing & prediction
│   └── schemas.py          # API request/response schemas
│
├── artifacts/
│   ├── model.keras         # Trained Neural Network
│   ├── scaler.pkl          # Feature scaler
│   ├── feature_columns.pkl # Training feature order
│   └── model_metadata.json # Model metadata
│
├── streamlit/
│   ├── streamlit_app.py    # Streamlit application
│   └── requirements.txt    # Streamlit dependencies
│
├── requirements-backend.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

---

## 🧠 Learnings

* Building an **end-to-end Neural Network regression solution**
* Importance of maintaining **training/inference preprocessing consistency**
* Serializing and loading machine learning artifacts
* Building production-style **REST APIs using FastAPI**
* Implementing request validation using **Pydantic**
* Validating model and preprocessing artifact compatibility
* Comparing independent inference pipelines for **prediction parity**
* Building an interactive ML application using **Streamlit**
* Deploying a machine learning application using **Streamlit Community Cloud**

---

## 📌 Key Takeaway

This project demonstrates the transition from:

> **Training a machine learning model**

to:

> **Building and deploying a complete machine learning application.**

The solution combines **Machine Learning + API Development + Software Engineering + Deployment** into a single end-to-end project.

---

## 📬 Contact

**Prathamesh Kamble**

Data Scientist
Mumbai, India

Email: [prathameshkamble99@gmail.com](mailto:prathameshkamble99@gmail.com)
