import tkinter as tk
from tkinter import messagebox
import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
def check_news():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some news text.")
        return
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    result = "Fake News ❌" if prediction == 0 else "Real News ✅"
    result_label.config(text=result)
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("400x300")
label = tk.Label(root, text="Enter News Text:", font=("Arial", 14))
label.pack(pady=10)
entry = tk.Text(root, height=5, width=40)
entry.pack(pady=10)
check_button = tk.Button(root, text="Check News", command=check_news)
check_button.pack(pady=5)
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)
root.mainloop()
