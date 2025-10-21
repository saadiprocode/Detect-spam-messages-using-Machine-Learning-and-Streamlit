import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()

def load_data(path):
    """Load the SMS Spam dataset."""
    df = pd.read_csv(path, sep='\t', names=['label', 'message'])
    return df

def clean_text(text):
    """Clean a single message."""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

def prepare_data(df):
    """Clean messages and return features and labels."""
    corpus = [clean_text(msg) for msg in df['message']]
    X = corpus
    y = pd.get_dummies(df['label'], drop_first=True).values
    return X, y
