## 📩 SMS Spam Detector

### *Detect spam messages using Machine Learning and Streamlit*

---

### 🌟 Overview

This project is a **machine learning-based SMS Spam Detector** that classifies text messages as **Spam** or **Not Spam**.
It uses **TF-IDF vectorization** and a **Naive Bayes classifier**, deployed with an interactive **Streamlit web app** featuring sample messages, a clean UI, and dark/light theme support.



### 🧠 Features

* 🧩 Machine Learning model trained on SMS Spam Collection dataset
* 🔤 Text preprocessing with stemming and stopword removal
* 🧮 TF-IDF vectorization for feature extraction
* 🤖 Naive Bayes classification
* 🌗 Dark & Light mode toggle
* 💬 Preloaded sample SMS messages for quick testing
* ✨ Elegant modern UI (glassmorphism style) built with Streamlit



### 🏗️ Project Structure


smsspamclassify/
│
├── app/
│   └── streamlit_app.py           # Streamlit web application
│
├── src/
│   ├── preprocess.py              # Data loading and cleaning
│   ├── train.py                   # Model training and saving
│   ├── utils.py                   # Helper functions to load model/vectorizer
│
├── data/
│   └── SMSSpamCollection.txt      # Dataset (tab-separated)
│
├── models/
│   ├── best_model.joblib          # Trained Naive Bayes model
│   └── tfidf_vectorizer.joblib    # TF-IDF vectorizer
│
├── venv/                          # Virtual environment (optional)
│
└── README.md                      # Project documentation




### ⚙️ Installation & Setup

#### 1️⃣ Clone this repository


git clone https://github.com/<your-username>/smsspamclassify.git
cd smsspamclassify


#### 2️⃣ Create and activate a virtual environment


python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On macOS/Linux


#### 3️⃣ Install dependencies


pip install -r requirements.txt



### 🧠 Train the Model

Run the training script to generate your model and vectorizer files:

python src/train.py


This will create:

* `models/best_model.joblib`
* `models/tfidf_vectorizer.joblib`



### 🚀 Run the Streamlit App

Launch the app using:


streamlit run app/streamlit_app.py


Then open your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**


### 📊 Model Details

| Component          | Description                                   |
| ------------------ | --------------------------------------------- |
| Algorithm          | Multinomial Naive Bayes                       |
| Feature Extraction | TF-IDF Vectorizer                             |
| Dataset            | SMS Spam Collection (UCI ML Repository)       |
| Accuracy           | ~98% on test data                             |
| Libraries          | scikit-learn, pandas, nltk, streamlit, joblib |



### 🎨 UI Preview

> The app features a sleek, modern interface with:

* Dark/Light mode toggle
* Glass-style message input box
* Real-time classification results
* Predefined sample messages



### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.



### 🧾 License

This project is licensed under the **MIT License** — feel free to use and modify it.



### 👨‍💻 Author

**Muhammad Saad**
📧 saad2542@gmail.com

