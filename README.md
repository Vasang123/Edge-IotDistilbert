# Edge-IoTDistilBERT: Fine-Tuning DistilBERT for Multi-Class Classification of Network Packet Attacks

## 📌 Overview
Edge-IoTDistilBERT is a **fine-tuned DistilBERT model** designed for **multi-class classification of network packet attacks** in **IoT environments**. This project aims to provide an efficient cybersecurity solution for resource-constrained devices by converting **PCAP data to text** and leveraging **NLP-based classification**.

This model is trained on the **Edge-IIoT dataset**, which represents multiple types of network attacks. 

## 🛠️ Key Features
- **Fine-Tuned DistilBERT Model**: Trained on **Edge-IIoT dataset** for **multi-class classification**.
- **PCAP-to-Text Preprocessing**: Converts network packet data into a textual format for NLP-based classification.
- **High Performance**: Achieves up to **99.99% accuracy** on both **80-20 and 70-30 dataset splits**.

## 📂 Repository Structure
This repository contains detailed documentation of each step taken in the research:

```
📁 Edge-IoTDistilBERT
│── 📜 README.md               # Project Overview
│── 📁 code/                    # Python code
│   ├── 📁 1. Dataset Preparation
│   │   ├── 📜 dataset_preparation.md      # PCAP-to-Text Preprocessing Details
│   │   ├── 📙 1.1. generate_textual_embedding.ipynb
│   │   ├── 📙 1.2. dataset_splitting.ipynb
│   ├── 📁 2. Model Training
│   │   ├── 📜 model_training.md    # Fine-Tuning DistilBERT for Classification
│   │   ├── 📙 2.1. model_training.ipynb
│   ├── 📁 3. Evaluation
│   │   ├── 📜 evaluation.md        # Performance Metrics & Comparison
│   │   ├── 📙 3.1. result.ipynb
│── 📁 data/                    # Dataset

```

## 🚀 Steps Covered in This Documentation
### 1️⃣ **Dataset & Preprocessing**
- Overview of the **Edge-IIoT dataset**
- **PCAP-to-text conversion** for NLP processing
- Label mapping and dataset splitting (**80-20 and 70-30 splits**)

### 2️⃣ **Fine-Tuning DistilBERT**
- Model selection: **distilbert-base-uncased**
- Tokenization and training process
- Optimization techniques (weight decay, batch processing)

### 3️⃣ **Model Evaluation & Results**
- **Precision, Recall, F1-Score, and Accuracy**
- **Confusion Matrix Analysis**
- Comparison with **other cybersecurity models** (e.g., SecurityBERT, CNN-LSTM)

### 4️⃣ **Future Improvements**
- Plans to **quantize the model** for resource-efficient deployment on **edge devices**

## 📌 Next Steps
- **Deploy on Edge Devices**: Optimize for real-world implementation.
- **Expand Dataset**: Improve model generalization with additional datasets.
- **Integrate with Threat Detection Systems**: Use in **SOC** environments for **real-time cybersecurity monitoring**.

## ⚠️ Disclaimer
- Some steps in this documentation might be **ineffective** due to the transition from a **Linux-based CPU environment** to a **Windows-based GPU environment**.
- This project might not contain the **full source code**, as parts have been omitted to improve clarity in the documentation.

## 🚧 Limitations
- Potential Overfitting: The model achieves *99.99% accuracy,* which may indicate memorization rather than generalization. Further testing on external datasets is needed.
- Dataset Bias: The dataset may contain highly distinguishable attack patterns, potentially affecting real-world performance.
- Reproducibility Concerns: Results depend on specific preprocessing steps, hyperparameters, and dataset versions. Ensure consistency when replicating.
- Limited Real-World Testing: The model has been validated on Edge-IIotset, but extensive real-world testing is required before deployment.

## 📧 Contact
For questions or contributions, reach out to:

📩 Valentino Setiawan - *valentino.setiawan@binus.ac.id* / *valentinosetiawan32@gmail.com*