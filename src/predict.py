import joblib 
from pathlib import Path
# This file enables us to take an email text from user and predict wether it is ham or spam 

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT/"model"
MODEL_PATH = MODEL_DIR/ "spam_tfidf_loreg.joblib"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found in {MODEL_PATH}. Run train.py to create it"
        )
    
    model = joblib.load(MODEL_PATH)

    print("Paste the email content you wish to enter: ")

    text = input(">").strip()

    if not text:
        print("Empty input.Please paste an email text.")
        return 
    
    prob_spam = model.predict_proba([text])[0][1]
    pred = 1 if prob_spam > 0.5 else 0

    label = "SPAM" if pred == 1 else "HAM"
    print(f"Prediction: {label}")
    print(f"Spam probability: {prob_spam:.4f}")

if __name__ == "__main__":
    main()