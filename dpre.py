import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

def data_preprocessing(df):
    df.dropna(subset=['CouncilArea'], inplace=True)
    imputer = SimpleImputer(strategy='most_frequent')
    df['YearBuilt'] = imputer.fit_transform(df[['YearBuilt']])
    imputer = SimpleImputer(strategy='mean')
    df['BuildingArea'] = imputer.fit_transform(df[['BuildingArea']])
    df = df.drop('Address', axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop(columns=['Date'], axis=1, inplace=True)
    df['Price_ln'] = df['Price'].apply(np.log)
    df = df.drop('Price', axis=1)

    # Save the resulting dataframe as a new CSV file
    df.to_csv("/home/project_docker/res_dpre.csv", index=False)

if __name__ == "__main__":
    # Load the dataset
    dataset_path = "/home/project_docker/loaded_dataset.csv"
    dataset = pd.read_csv(dataset_path)
    data_preprocessing(dataset)
    print("Data preprocessing completed!")
    # Invoke the next Python file
    exec(open("/home/project_docker/eda.py").read())