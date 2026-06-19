# 🏦 Banking ChatBot

A machine learning–based Banking ChatBot built with **Python**, **Scikit-learn**, **Natural Language Processing (NLP)**, and **Gradio**.

The chatbot is trained on a custom banking dataset to understand common customer queries and respond with relevant banking information using intent classification.

**Developed by:** **Mahesh Pawar**
**GitHub:** https://github.com/MR1229

---

## 📖 About the Project

This project demonstrates how traditional machine learning techniques can be used to build a conversational banking assistant without relying on external AI APIs or large language models.

Instead of generating responses freely, the chatbot analyzes user input, predicts the most appropriate banking intent, and returns a corresponding response learned from its training dataset.

The project was created for learning, experimentation, and portfolio purposes, with a focus on NLP, text classification, and practical chatbot development.

---

## ✨ Key Features

* 🤖 Machine Learning–based chatbot
* 💬 Interactive web interface built with Gradio
* 🏦 Handles common banking-related queries
* 📊 TF-IDF vectorization for text processing
* 🧠 Multinomial Naive Bayes intent classification
* 🛡️ Confidence-based fallback for unsupported questions
* 📚 Easily expandable training dataset
* ⚡ Automatic model training when required

---

## 🛠️ Technology Stack

| Component         | Technology              |
| ----------------- | ----------------------- |
| Language          | Python                  |
| Machine Learning  | Scikit-learn            |
| NLP               | TF-IDF + NLTK           |
| Model             | Multinomial Naive Bayes |
| Data Handling     | Pandas                  |
| User Interface    | Gradio                  |
| Model Persistence | Joblib                  |

---

## 📂 Repository Structure

```text
Banking-ChatBot/
│
├── banking_chatbot.py
├── banking_support.csv
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/MR1229/Banking-ChatBot.git
cd Banking-ChatBot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the chatbot

```bash
python banking_chatbot.py
```

If trained model files are not already available, the application will automatically train a new model using the dataset before launching the chat interface.

---

## 💡 Example Queries

* How do I check my account balance?
* How can I block my debit card?
* How do I transfer money using UPI?
* How do I open a savings account?
* What documents are required for KYC?
* How can I apply for a personal loan?
* What is the IFSC code?
* How do I reset my internet banking password?

---

## 🔄 Updating the Dataset

Improving the chatbot is straightforward:

1. Open `banking_support.csv`
2. Add or edit banking questions and responses
3. Save the dataset
4. Run:

```bash
python banking_chatbot.py
```

The model will automatically retrain using the updated data.

---

## ⚙️ How It Works

```text
User Input
     │
     ▼
Text Cleaning & Preprocessing
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Multinomial Naive Bayes Model
     │
     ▼
Intent Prediction
     │
     ▼
Response Selection
     │
     ▼
Chat Interface (Gradio)
```

---

## 📌 Future Enhancements

Potential improvements for future versions include:

* Transformer-based NLP models
* Semantic search using embeddings
* Multilingual support
* Voice-based interaction
* Retrieval-Augmented Generation (RAG)
* Integration with real banking APIs
* User authentication and personalized assistance

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.

It is **not affiliated with any real bank or financial institution** and does not access live customer accounts, transactions, or banking systems.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Mahesh Pawar**

GitHub: **https://github.com/MR1229**

If you found this project useful, consider giving it a ⭐ on GitHub.
