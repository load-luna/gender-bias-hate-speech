import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import numpy as np


def load_data_from_config(config: dict) -> DatasetDict:
    dataset_config = config.get("dataset", {})
    source = dataset_config.get("source")

    if source == "kaggle":
        kaggle_config = dataset_config.get("kaggle", {})

        data_path = kaggle_config.get("data_path")
        file_path = kaggle_config.get("file_path")

        if not data_path:
            raise ValueError(
                "Missing dataset.kaggle.data_path in config."
            )

        if not file_path:
            raise ValueError(
                "Missing dataset.kaggle.file_path in config."
            )
        
        return load_data_from_Kaggle(data_path, file_path)
    elif source == "local":
        local_config = dataset_config.get("local", {})
        path = local_config.get("path")

        if not path:
            raise ValueError(
                "Missing dataset.local.path in config."
            )
        df = pd.read_csv(path)

        return Dataset.from_pandas(
            df,
            preserve_index=False,
        )
    else:
        raise ValueError(f"Unknown data source: {source}. Please choose 'kaggle' or 'local'.")


#"mahmoudabusaqer/combined-hate-speech-dataset", "3DatasetsCombined.csv"
#"eldrich/hate-speech-offensive-tweets-by-davidson-et-al", "data/labeled_data.csv"
def load_data_from_Kaggle(data_path: str, file_path: str) -> DatasetDict:

    return kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            data_path,
            file_path,
        )

def prepare_datasetold(df0: DatasetDict, dataset_name: str) -> DatasetDict:
    if dataset_name == "davidson":

        
        df0['class'] = df0['class'].map({0: 1, 1: 1, 2: 0})
        c=df0['class']
        df0.rename(columns={'tweet' : 'text',
                            'class' : 'category'}, 
                            inplace=True)
        a=df0['text']
        b=df0['category'].map({0: 'neither', 1: 'hate_speech'})

        df= pd.concat([a,b,c], axis=1)
        df.rename(columns={'class' : 'label'}, 
                            inplace=True)
        
        return df
    elif dataset_name == "combined":


        c=df0['class']
        df0.rename(columns={'tweet' : 'text',
                            'class' : 'category'}, 
                            inplace=True)
        a=df0['text']
        b=df0['category'].map({0: 'neither', 1: 'hate_speech'})

        df= pd.concat([a,b,c], axis=1)
        df.rename(columns={'class' : 'label'}, 
                            inplace=True)
        return df


    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Please choose 'davidson' or 'combined'.")





def prepare_example(
    example: dict,
    text_column: str,
    label_column: str,
    mapping: dict,
    names: dict,
) -> dict:
    original_label = example[label_column]
    new_label = mapping[original_label]

    return {
        "text": example[text_column],
        "label": new_label,
        "category": names[new_label],
    }


def prepare_dataset(
    dataset: Dataset | DatasetDict,
    config: dict,
) -> Dataset | DatasetDict:

    # 1. Read config
    dataset_config = config["dataset"]

    text_column = dataset_config["columns"]["text"]
    label_column = dataset_config["columns"]["label"]

    mapping = {
        int(key): int(value)
        for key, value in dataset_config["label_mapping"]["mapping"].items()
    }

    names = {
        int(key): value
        for key, value in dataset_config["label_mapping"]["names"].items()
    }

    # Prepare DatasetDict
    if isinstance(dataset, DatasetDict):
        prepared_splits = {}

        for split_name, split in dataset.items():
            prepared_splits[split_name] = split.map(
                lambda example: prepare_example(
                    example,
                    text_column,
                    label_column,
                    mapping,
                    names,
                ),
                remove_columns=split.column_names,
            )

        return DatasetDict(prepared_splits)

    #  Prepare single Dataset
    if isinstance(dataset, Dataset):
        return dataset.map(
            lambda example: prepare_example(
                example,
                text_column,
                label_column,
                mapping,
                names,
            ),
            remove_columns=dataset.column_names,
        )

    raise TypeError(
        "dataset must be a Hugging Face Dataset or DatasetDict."
    )

def print_label_distribution(df: pd.DataFrame):
    neither,hate  = np.bincount(df['label'])
    total = hate + neither
    print('Examples:\n    Total: {}\n    hate: {} ({:.2f}% of total)\n    Neither: {} ({:.2f}% of total)\n'.format(
        total, hate, 100 * hate / total, neither, 100 * neither / total))

def filter_max_sequence_length(tokenizer, dataset):
    """Filter out sequences that exceed the maximum length for a given tokenizer."""
    max_length = tokenizer.model_max_length

    def filter_function(example):
        return len(tokenizer(example["text"])["input_ids"]) <= max_length

    filtered_dataset = dataset.filter(filter_function)
    return filtered_dataset



def split_dataset(dataset: Dataset, config: dict) -> DatasetDict:
    """Split the dataset into train, validation, and test sets based on the provided configuration."""
    split_config = config.get("dataset", {}).get("split", {})
    train_size = split_config.get("train_split_size", 0.8)
    val_size = split_config.get("validation_split_size", 0.1)
    test_size = split_config.get("test_split_size", 0.1)

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    # Shuffle the dataset before splitting
    shuffled_dataset = dataset.shuffle(seed=42)

    # Calculate the number of examples for each split
    total_examples = len(shuffled_dataset)
    train_end = int(total_examples * train_size)
    val_end = train_end + int(total_examples * val_size)

    # Split the dataset
    train_dataset = shuffled_dataset.select(range(train_end))
    val_dataset = shuffled_dataset.select(range(train_end, val_end))
    test_dataset = shuffled_dataset.select(range(val_end, total_examples))

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    