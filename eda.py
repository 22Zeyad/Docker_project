import sys
import pandas as pd

# Function to perform exploratory data analysis
def exploratory_data_analysis(df, output_file):
    # Redirect stdout to the output file
    sys.stdout = open(output_file, 'w')

    print(df.info())
    print(df.describe())
    print(df.isna().sum())

    # Reset stdout to the default (console)
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    # Load the preprocessed dataset
    df = pd.read_csv("/home/project_docker/res_dpre.csv")
    
    # Specify the output file for redirected prints
    output_file = "/home/project_docker/eda-in-1.txt"
    
    # Perform exploratory data analysis and save prints to the file
    exploratory_data_analysis(df, output_file)
    
    print("Exploratory data analysis completed!")
    
    # Invoke the next Python file
    exec(open("/home/project_docker/vis.py").read())