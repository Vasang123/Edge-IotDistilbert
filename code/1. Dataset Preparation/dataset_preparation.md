# 📜 Dataset Preparation: PCAP-to-Text Preprocessing & Dataset Splitting

## 📌 Overview
This document outlines the process of converting **DNN-EdgeIIoT-dataset.csv** into a text format suitable for training the **Edge-IoTDistilBERT** model. Additionally, it details how the dataset is split for training and evaluation to ensure a balanced approach.

---

## 🗂️ Step 1: Converting PCAP to Text
Since network traffic is stored in raw CSV format, this step will transform the CSV file into a single paragraph containing all relevant information for each network transaction.

### 🔹 **Preprocessing Steps**:
1️⃣ **Extract Packet Information**
   - Load all network packets from **DNN-EdgeIIoT-dataset.csv** using pandas.
   ```python
   import pandas as pd

   # Load dataset
   df = pd.read_csv("DNN-EdgeIIoT-dataset.csv")
   
   ```

2️⃣ **Format Data as Text**
   - Convert each network event into structured text descriptions.
   - Example format:
     ```plaintext
     At timestamp 12:34:56, the source IP 192.168.0.1 sent a TCP SYN packet to destination IP 192.168.1.1 on port 80.....
     ```
   - Python code to generate text embeddings:
     ```python
     def generate_textual_embedding(row):
         return f"At timestamp {row['timestamp']}, the source IP {row['src_ip']} sent a {row['protocol']} packet to destination IP {row['dst_ip']} on port {row['dst_port']} with packet size {row['packet_size']}..."
     
     df['text_embedding'] = df.apply(generate_textual_embedding, axis=1)
     ```

3️⃣ **Save into JSON**
   - Save the generated text into a JSON file for later processing.
   ```python
   df[['text_embedding', 'label']].to_json("processed_data.json", orient="records", lines=True)
   ```

---

## ✂️ Step 2: Dataset Splitting
To evaluate the model's robustness, the dataset is split into **training and validation sets** with two configurations:

| Split Type | Training (%) | Testing (%) |
|------------|-------------|-------------|
| **80-20 Split** | 80% | 20% |
| **70-30 Split** | 70% | 30% |


---

## 📌 Summary
✅ **CSV files are converted into JSON with all network transaction details.**  
✅ **Textual embeddings are generated for NLP-based processing.**  
✅ **Dataset is split into 80-20 and 70-30 configurations to test model generalization.**  

📢 _For further improvements, testing on real-world network traffic is recommended!_ 🚀
