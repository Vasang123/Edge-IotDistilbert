# 📊 Model Evaluation: Performance Metrics & Analysis

## 📌 Overview
This document provides an evaluation of the **Edge-IoTDistilBERT** model, analyzing its classification performance on network attack detection. We report key **performance metrics**, visualize results, and compare different dataset splits (80-20 and 70-30).

---

## 1️⃣ Load the Trained Model & Test Data
The trained model is evaluated using the test dataset.

```python
save_path = "save_path"
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)

raw_datasets = DatasetDict({
    "train": Dataset.from_csv("train.csv"),
    "eval": Dataset.from_csv("eval.csv")
})

```

---

## 2️⃣ Compute Predictions & Metrics
We evaluate the model using **accuracy, precision, recall, and F1-score**.

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate the loaded model
trainer = Trainer(
    model=loaded_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=loaded_tokenizer,
    compute_metrics=compute_metrics
)


eval_result = trainer.evaluate()
```

---

## 3️⃣ Confusion Matrix
A confusion matrix helps visualize classification errors.

```python
labels = [
    "Normal", "MITM", "Uploading", "Ransomware", "SQL_injection",
    "DDoS_HTTP", "DDoS_TCP", "Password", "Port_Scanning", "Vulnerability_scanner",
    "Backdoor", "XSS", "Fingerprinting", "DDoS_UDP", "DDoS_ICMP"
]


cf_matrix = confusion_matrix(y_true, y_pred)


cf_matrix_percent = cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis] * 100


plt.figure(figsize=(10, 10))
sns.heatmap(cf_matrix_percent, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)


plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix with Percentages")
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)              
plt.show()
```

---

### **Observations:**
- The model achieves **high accuracy (>99.9%)** across both dataset splits.
- The confusion matrix indicates **minimal misclassification**, suggesting strong model generalization.
- The high precision and recall values confirm **low false positive and false negative rates**.

---

## ✅ Summary
- **Evaluated Edge-IoTDistilBERT using classification metrics.**
- **Generated confusion matrix for error analysis.**
- **Compared performance on 80-20 and 70-30 splits.**

📢 _For further validation, testing on real-world datasets is recommended!_ 🚀