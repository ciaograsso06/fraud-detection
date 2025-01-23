import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo e retorna métricas e matriz de confusão.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return report, conf_matrix

if __name__ == "__main__":
    DATA_PATH = "./data/processed/creditcard_2023.csv"
    MODEL_PATH = "./models/fraud_model.pkl"
    REPORT_PATH = "./outputs/report.txt"
    CONF_MATRIX_PATH = "./outputs/confusion_matrix.txt"

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    model = joblib.load(MODEL_PATH)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    report, conf_matrix = evaluate_model(model, X_test, y_test)

    with open(REPORT_PATH, "w") as report_file:
        report_file.write(report)
    with open(CONF_MATRIX_PATH, "w") as conf_file:
        conf_file.write(str(conf_matrix))
    
    print("Relatório e matriz de confusão salvos em ./outputs/")
