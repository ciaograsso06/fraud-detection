import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(input_file):
    """
    Carrega o dataset processado.
    """
    return pd.read_csv(input_file)

def train_model(X_train, y_train):
    """
    Treina o modelo de classificação.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    DATA_PATH = "./data/processed/creditcard_2023.csv"
    MODEL_PATH = "./models/fraud_model.pkl"

    df = load_data(DATA_PATH)
    X = df.drop("Class", axis=1)  # 'Class' é a variável alvo
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
