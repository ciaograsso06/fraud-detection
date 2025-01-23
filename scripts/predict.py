import pandas as pd
import joblib

def predict(model, input_data):
    """
    Faz previsões usando o modelo treinado.
    """
    return model.predict(input_data)

if __name__ == "__main__":
    MODEL_PATH = "./models/fraud_model.pkl"
    INPUT_FILE = "./data/processed/sample_input.csv" 
    OUTPUT_FILE = "./outputs/predictions.txt"

    model = joblib.load(MODEL_PATH)
    input_data = pd.read_csv(INPUT_FILE)

    predictions = predict(model, input_data)

    with open(OUTPUT_FILE, "w") as output_file:
        output_file.write("\n".join(map(str, predictions)))

    print(f"Previsões salvas em: {OUTPUT_FILE}")
