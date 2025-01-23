import os
import shutil
import kagglehub

def download_and_move_dataset(kaggle_dataset, target_folder):
    """
    Faz o download do dataset usando KaggleHub e move os arquivos para outra pasta.
    """
    path = kagglehub.dataset_download(kaggle_dataset)
    print("Dataset baixado em:", path)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(path):
        src_file = os.path.join(path, filename)
        dest_file = os.path.join(target_folder, filename)
        shutil.move(src_file, dest_file)
    
    print(f"Arquivos movidos para: {target_folder}")

if __name__ == "__main__":
    KAGGLE_DATASET = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
    
    TARGET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

    download_and_move_dataset(KAGGLE_DATASET, TARGET_FOLDER)