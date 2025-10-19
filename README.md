# 🧠 Consumer Complaint Classification using Machine Learning

This project trains a machine learning model to classify consumer complaints into product categories based on their text descriptions. It uses TF-IDF vectorization and Logistic Regression to build the classifier.

---

## ✅ Features

* Text preprocessing (cleaning + stopword removal)
* TF-IDF vectorization
* Label encoding
* Logistic Regression model
* Accuracy evaluation + classification report
* Confusion matrix visualization
* Feature importance graph
* User input predictions
* Sample text predictions

---

## 📂 Dataset

The script expects the dataset at:

```
/kaggle/input/consumer-complaint-database/rows.csv
```

Make sure the file exists at that location, or update the `path` variable in the code.

Only the following columns are used:

* `Product`
* `Consumer complaint narrative`

Rows with missing values in these fields are removed automatically.

---

## 🛠️ Installation & Requirements

Install the required libraries before running the script:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

For NLTK stopwords, include this before preprocessing:

```python
import nltk
nltk.download('stopwords')
```

---

## 🚀 How to Run

1. Place the dataset at the correct path or update the script.
2. Run the Python script normally:

```bash
python your_script_name.py
```

3. After training and evaluation, the script will:

   * Show accuracy metrics
   * Display the confusion matrix
   * Plot most important words
   * Ask for user input to predict a product category

---

## 🧾 Predictions

Sample user inputs are classified at the end of the script:

```text
"The bank closed my account without any reason."  
→ Product: Bank account or service

"My mortgage payment was not processed on time."  
→ Product: Mortgage
```

You can also enter your own text when prompted.

---

## 📊 Output Includes

* Model accuracy %
* Classification report
* Confusion matrix (visual)
* Top 10 most important words
* User and sample predictions

---

## 🔄 Modifying Dataset Path


## screenshots
![my local image] {11.png}  
![my local image] {12.png}  
![my local image] {13.png}  

Change this line if needed:

```python
path = "/kaggle/input/consumer-complaint-database/rows.csv"
```

Replace it with a local or absolute path.

---

## ✅ Project File Structure (Suggested)

```
├── README.md
├── model.py        # Your full script
└── /data
    └── rows.csv
```

---

## 📌 Final Notes

* Make sure the dataset exists before running.
* NLTK stopwords must be downloaded once.
* Works locally, in Kaggle, or Jupyter.
