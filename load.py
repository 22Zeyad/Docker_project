import pandas as pd

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__== "__main__":
    dataset_path = "/home/project_docker/data.csv"
    df = load_dataset(dataset_path)
    df.to_csv("/home/project_docker/loaded_dataset.csv", index=False)
    print("Dataset loaded successfully!")
    # Invoke the next Python file
    exec(open("/home/project_docker/dpre.py").read())