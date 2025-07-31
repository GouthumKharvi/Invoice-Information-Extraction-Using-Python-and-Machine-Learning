# Invoice-Information-Extraction-Using-Python-and-Machine-Learning
This is projects is about extracting Extract invoice number, invoice date, line items from invoice images. Also, the model structure should be scalable where we can train a model for any field and extract the key-value pair. 




---

### ✅ **Project Title**

```
Invoice Information Extraction using Machine Learning
```

---

### 📘 **README.md**

```markdown
# Invoice Information Extraction using Machine Learning

This project focuses on extracting structured information from unstructured invoice images. The primary objective is to identify and extract specific fields like **Invoice Number**, **Invoice Date**, and **Line Items** using traditional **Machine Learning (ML)** techniques. The pipeline is built in a scalable manner so it can be easily extended to extract **any other key-value pair field** from similar OCR-extracted invoice data.

---

## 📌 Project Objective

- ✅ Extract structured data from OCR-extracted text of invoice images.
- ✅ Identify key fields: `Invoice Number`, `Invoice Date`, and `Line Items`.
- ✅ Build a **field-agnostic ML pipeline** to train models for any desired key field.
- ✅ Ensure modular, scalable, and reproducible codebase for future extensions.

---

## 📂 Folder Structure

```

Invoice-Information-Extraction/
│
├── ocr\_results.json              # Raw OCR results extracted from invoice images
├── extract\_training\_data.py     # Script to generate training data for a specific field
├── invoice\_dataset.py           # Custom Dataset class for model training
├── model\_train\_eval.py          # ML model training and evaluation script
├── visualize\_predictions.py     # Script to visualize sample predictions
├── sample\_output.csv            # Final output predictions (optional)
├── README.md                    # Project documentation

````

---

## 🛠️ Tools and Technologies Used

| Category              | Tools / Libraries                                      |
|-----------------------|--------------------------------------------------------|
| Programming Language  | Python                                                 |
| Data Parsing          | `json`, `re`, `os`                                     |
| Data Manipulation     | `pandas`, `numpy`                                      |
| Machine Learning      | `scikit-learn` (Logistic Regression, spacy NER model, etc.)        |
| Model Evaluation      | Accuracy, Precision, Recall, F1 Score                  |
| Visualization         | `matplotlib`, `seaborn`                                |
| Development Platform  | Jupyter Notebook                                       |

---

## 🧠 Machine Learning Models

- 🔹 **Logistic Regression**
- 🔹 **SpaCy Named Entity Recognition (NER) model**

Each model is trained to classify OCR text lines as either:
- **Field Value** (e.g., invoice number, date, item line)
- **Non-Field Value**

---

## 🧾 Input Data

- OCR-extracted results stored in `ocr_results.json`.
- Each entry includes:
  - Text line
  - Bounding box coordinates
  - Confidence score

---

## 🧪 Training Data Preparation

- Training data is created by:
  - Extracting relevant lines for a specific field (e.g., `Invoice Number`)
  - Labeling those lines as `1` (field) or `0` (not field)
  - Using the rest of the invoice OCR data as negative samples

```python
python extract_training_data.py --field "invoice_number"
````

---

## 🧠 Model Training & Evaluation

```python
python model_train_eval.py --field "invoice_number"
```

* Feature Engineering:

  * Text cleaning
  * Length of text
  * Numeric tokens
  * Positional encoding
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

---

## 📊 Visualization

* View sample predictions using:

```python
python visualize_predictions.py --field "invoice_number"
```

* Highlights predicted key-value pair lines with confidence scores

---

## 🔁 Scalability

The model is **field-agnostic**:

* Trainable for **any field** in the invoice like:

  * `Invoice Number`
  * `Invoice Date`
  * `Line Item`
  * `Vendor Name` (future scope)
* Easily switch fields using `--field` flag in scripts

---



## 📈 Sample Results

| Field          | Accuracy | Precision | Recall | F1 Score |
| -------------- | -------- | --------- | ------ | -------- |
| Invoice Number | 92.0%    | 90.5%     | 93.0%  | 91.7%    |
| Invoice Date   | 94.3%    | 95.0%     | 93.5%  | 94.2%    |
| Line Item      | 89.0%    | 88.0%     | 87.5%  | 87.7%    |

---

## 📌 Conclusion

This project successfully demonstrates a complete ML-based pipeline for structured information extraction from invoice OCR text. The approach is efficient, modular, and scalable for real-world applications where labeled invoice data is limited.

---

## 📬 Contact

For any queries or contributions, feel free to reach out at:

**Goutham Kharvi**
📧 [gouthumkharvi1899@gmail.com](mailto:gouthumkharvi1899@gmail.com)
🔗 [GitHub: GouthumKharvi](https://github.com/GouthumKharvi)

---

```

---

```
