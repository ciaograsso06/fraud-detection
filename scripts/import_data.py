import kaggle
import os

def download_dataset(kaggle_dataset, output_dir):
    """
    Faz o download do dataset do Kaggle.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    kaggle.api.dataset_download_files(kaggle_dataset, path=output_dir, unzip=True)
    print(f"Dataset baixado em: {output_dir}")

if __name__ == "__main__":
    KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
    OUTPUT_DIR = "./data/raw/"
    
    download_dataset(KAGGLE_DATASET, OUTPUT_DIR)
