from datasets import load_dataset
import json
from tqdm import tqdm
import pandas as pd




def download_data(base_url, num_shards):
    # Download the data
    print("Downloading data...")
    urls = [base_url.format(i=i) for i in range(num_shards)]
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    return dataset

def download_data_from_kaggle(path):

    # Download latest version
    print("Downloading data...")
    path = kagglehub.dataset_download("manann/quotes-500k")

    print("Path to dataset files:", path)

    return path
    


def extract_prompts(dataset, jsonl_file_path):
    # Write data to the jsonl file
    prompts = {}
    print('Extracting data to:', jsonl_file_path)

    with open(jsonl_file_path, 'w') as f:
        with tqdm(desc="Processing prompts", unit=" prompt") as pbar:
            for index, row in enumerate(dataset):
                prompts[index] = row['json']['prompt']
                f.write(json.dumps(prompts[index]) + '\n')
                
                pbar.update(1)




def read_data_from_csv(csv_path):
    # Read data from the jsonl file
    df = pd.read_csv(csv_path)
    df['quote']
    quotes = df['quote'].tolist()
    
    return quotes


def load_quotes_from_csv(file_path):
    print('Loading quotes from:', file_path)
    prompts = []
    quotes_df = pd.read_csv(file_path)
    quotes_df['quote'] = quotes_df['quote'] + quotes_df['author'].apply(lambda x: f" - {x}" if pd.notna(x) else "")
    quotes = quotes_df["quote"].astype(str).tolist()
    print("Quotes loaded:", len(quotes))   # should be 499709
    print("First quote:", quotes[0][:100])
    print("Data loaded successfully.")
    return quotes[:10000]


if __name__ == "__main__":
    csv_file_path = r"C:\Users\jov2bg\Desktop\PromptSearch\search_engine\data\quotes_new.csv"
    download_data_from_kaggle(csv_file_path)
    read_data_from_csv(csv_file_path)