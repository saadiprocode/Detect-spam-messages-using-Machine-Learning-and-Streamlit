from src.preprocess import load_data, prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os


def train_and_select(data_path='data/SMSSpamCollection.txt', model_dir='models'):
    # Load and preprocess data
    df = load_data(data_path)
    X, y = prepare_data(df)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=2500)
    X = vectorizer.fit_transform(X).toarray()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Training complete!\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'best_model.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))

    print(f"\n✅ Model and vectorizer saved successfully in '{model_dir}'")


if __name__ == "__main__":
    train_and_select()
