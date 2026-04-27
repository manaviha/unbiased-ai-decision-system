import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def analyze_and_train(file_path):

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            return {"error": "Unable to read CSV file"}

    # Clean column names
    df.columns = df.columns.str.strip().str.upper()

    # Rename columns automatically
    df.rename(columns={
        "INCOME(IN THOUSANDS)": "INCOME",
        "CREDIT SCORES": "CREDITSCORE",
        "CREDITSCOR": "CREDITSCORE"
    }, inplace=True)

    # Check Gender
    if "GENDER" not in df.columns:
        return {"error": "Missing GENDER column"}

    # Check Target
    if "APPROVED" in df.columns:
        target_col = "APPROVED"
    elif "TARGET" in df.columns:
        target_col = "TARGET"
    else:
        return {"error": "Missing APPROVED or TARGET column"}

    # Required Columns
    needed = ["AGE", "INCOME", "CREDITSCORE", "GENDER"]

    for col in needed:
        if col not in df.columns:
            return {"error": f"Missing required column: {col}"}

    df = df.dropna()

    # Gender Convert
    df["GENDER"] = df["GENDER"].astype(str).str.title()
    df["GENDER"] = df["GENDER"].map({"Male": 1, "Female": 0})

    # Target Convert
    df[target_col] = df[target_col].astype(str).str.title()
    df[target_col] = df[target_col].replace({
        "Yes": 1,
        "No": 0,
        "1": 1,
        "0": 0
    })

    df[target_col] = pd.to_numeric(df[target_col])

    # Features
    X = df[["AGE", "INCOME", "CREDITSCORE", "GENDER"]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    predictions = model.predict(X)
    df["PREDICTION"] = predictions

    result = df.groupby("GENDER")["PREDICTION"].mean()

    male_rate = result.get(1, 0)
    female_rate = result.get(0, 0)

    bias_score = abs(male_rate - female_rate)

    return {
        "accuracy": round(accuracy, 2),
        "male_rate": round(male_rate, 2),
        "female_rate": round(female_rate, 2),
        "bias_score": round(bias_score, 2)
    }