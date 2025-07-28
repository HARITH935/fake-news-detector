import pandas as pd
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")
fake_df["label"] = 0
real_df["label"] = 1
data = pd.concat([fake_df, real_df])
data = data.sample(frac=1).reset_index(drop=True)
print(data.head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = data[['text', 'label']]
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
def predict_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return "Fake News" if prediction[0] == 0 else "Real News"
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter news text to check (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the fake news detector. Goodbye!")
            break
        result = predict_news(user_input)
        print("Prediction:", result)
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")