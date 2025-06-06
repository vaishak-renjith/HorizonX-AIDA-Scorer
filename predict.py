import joblib

model = joblib.load("model/aida_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_aida(email_text):
    vec = vectorizer.transform([email_text])
    scores = model.predict(vec)[0]
    return {
        "attention": round(scores[0], 1),
        "interest": round(scores[1], 1),
        "desire": round(scores[2], 1),
        "action": round(scores[3], 1),
        "total": round(scores[0] + scores[1] + scores[2] + scores[3], 1)
    }

# Example usage
if __name__ == "__main__":
    email = input("Enter email text: ")
    print(predict_aida(email))
