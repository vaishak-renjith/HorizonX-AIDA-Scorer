import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Load data
df = pd.read_csv("data/leads_synthetic.csv")

# Preprocess the text
df["processed_body"] = df["body"].apply(preprocess_text)

X = df["processed_body"]
y = df[["attention_score", "interest_score", "desire_score", "action_score"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize with more parameters tuned
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,            # Ignore terms that appear in fewer than 2 documents
    max_df=0.8          # Ignore terms that appear in more than 80% of documents
)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model with regularization
model = MultiOutputRegressor(
    Ridge(alpha=1.0, solver='sag'),  # Try different alpha values
    n_jobs=-1
)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model/aida_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# Print some info
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print("Model training complete.")