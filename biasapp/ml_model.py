import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def analyze_and_train(file_path):
    df = pd.read_csv(file_path)

    # ✅ Check columns
    if "Gender" not in df.columns or "Approved" not in df.columns:
        return {"error": "Dataset must contain Gender and Approved"}

    # ✅ Convert categorical to numeric
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # ✅ Features and target
    X = df[["Gender"]]
    y = df["Approved"]

    # ✅ Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ✅ ML Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # ✅ Predictions for bias check
    predictions = model.predict(X)

    df["Prediction"] = predictions

    # ✅ Bias calculation
    result = df.groupby("Gender")["Prediction"].mean()

    male_rate = result.get(1, 0)
    female_rate = result.get(0, 0)

    bias_score = abs(male_rate - female_rate)

    return {
        "accuracy": round(accuracy, 2),
        "male_rate": round(male_rate, 2),
        "female_rate": round(female_rate, 2),
        "bias_score": round(bias_score, 2)
    }