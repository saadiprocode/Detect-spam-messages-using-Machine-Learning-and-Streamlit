
import joblib
import os

def load_vectorizer(model_dir='models'):
    """Load the saved TF-IDF vectorizer."""
    return joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))

def load_model(model_dir='models'):
    """Load the best trained spam classification model."""
    return joblib.load(os.path.join(model_dir, 'best_model.joblib'))
