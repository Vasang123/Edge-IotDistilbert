# 📙 Model Training: Fine-Tuning DistilBERT for Network Attack Classification

## 📌 Overview
This document details the process of **fine-tuning DistilBERT** for **multi-class classification** of network packet attacks using the **Edge-IIoT dataset**. The training process involves tokenizing the dataset, configuring hyperparameters, and training the model with evaluation metrics.

---

## 1️⃣ Load the Pre-Trained DistilBERT Model
DistilBERT is a lightweight Transformer model that retains most of BERT’s accuracy while improving efficiency.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "distilbert-base-uncased"
num_classes = 15  # Number of attack categories

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

---

## 2️⃣ Tokenization & Data Preparation
The raw dataset (converted from PCAP) is tokenized into a format suitable for DistilBERT.

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

---

## 3️⃣ Model Training Configuration
Hyperparameters are set to optimize model training.

```python
from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    "test-trainer", 
    num_train_epochs=1, 
    evaluation_strategy="epoch", 
    weight_decay=5e-4, 
    save_strategy="no", 
    report_to="none")


```

---

## 4️⃣ Train the Model
Define a trainer and initiate training.

```python
trainer = Trainer(
    classifier,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
```

---

## 5️⃣ Save the Trained Model
After training, the model is saved for inference.

```python
save_path = "save_folder" 
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

```

---

## ✅ Summary
- **Loaded DistilBERT and fine-tuned it on network attack data.**
- **Tokenized network traffic text for model input.**
- **Configured training with optimized hyperparameters.**
- **Trained and saved the final model for evaluation and deployment.**

📢 _For best performance, experiment with different learning rates and batch sizes!_ 🚀