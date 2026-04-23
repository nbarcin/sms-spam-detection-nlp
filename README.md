# 📩 SMS Spam Detection using NLP & Machine Learning

## 📌 Overview

This project focuses on building a high-performance SMS Spam Detection system using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify messages as **spam** or **ham (not spam)** based on their textual content.

The model achieves **~96% accuracy** using classical machine learning approaches combined with effective text preprocessing and feature engineering.

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains:

* **5,572 SMS messages**
* **2 columns:**

  * `class` → spam / ham
  * `text` → message content

Dataset link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## ⚙️ Technologies & Libraries

* Python
* pandas, numpy
* scikit-learn
* neattext (text preprocessing)
* TextBlob (sentiment analysis)

---

## 🔍 Project Pipeline

### 1. Data Cleaning & Preprocessing

* Removed unnecessary columns
* Cleaned text using `neattext`
* Combined text for analysis using:

```python
s = ''.join(df['text'])
```

---

### 2. Feature Engineering

* Applied **CountVectorizer**
* Used **n-grams (1,2)** to capture context

---

### 3. Model Building

#### 🌲 Random Forest Pipeline

```python
Pipeline([
 ('CountVect', CountVectorizer(ngram_range=(1,2))),
 ('Classifier', RandomForestClassifier())
])
```

#### ⚡ Support Vector Machine (SVM)

```python
Pipeline([
 ('CountVect', CountVectorizer(ngram_range=(1,2))),
 ('Classifier', SVC())
])
```

---

## 📈 Results

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 0.9668   |
| SVM           | 0.9668   |

✔ Both models achieved strong and similar performance.

---

## 💬 Sentiment Analysis

Used **TextBlob** to analyze sentiment polarity:

```python
from textblob import TextBlob
blob = TextBlob(text)
blob.sentiment.polarity
```

---

## 🧠 Named Entity Recognition (NER)

Basic exploration of Named Entity Recognition (NER) was performed to identify entities in text, though it was not used directly in model training.

---

## 🚀 Key Insights

* N-gram features significantly improve performance
* Classical ML models are highly effective for spam detection
* SVM and Random Forest perform similarly on this dataset
* Text preprocessing plays a critical role


