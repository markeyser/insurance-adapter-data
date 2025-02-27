"""
Module for validating and analyzing processed QA datasets.

This script loads CSV-formatted QA datasets, checks dataset balance,
computes token lengths using a BERT tokenizer, and plots token
length distributions. It also detects duplicate questions and
verifies dataset consistency.
"""

import csv
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# Load a BERT-like tokenizer
# Default: "bert-base-uncased". If fine-tuning DeepSeek-R1-Distill-Qwen-14B,
# replace with its tokenizer.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define output directory for images
IMG_DIR: Path = Path(__file__).resolve().parent.parent.parent / "docs/assets"
IMG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

def load_csv(file_path: str) -> List[Dict[str, str]]:
    """Loads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict[str, str]]: A list of dataset entries, where each entry
        is represented as a dictionary.
    """
    data: List[Dict[str, str]] = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["label"] = str(int(row["label"]))  # Ensure labels are valid
            data.append(row)
    return data

def get_token_length(text: str) -> int:
    """Tokenizes text and returns the number of tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        int: Number of tokens in the text.
    """
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def print_stats(name: str, lengths: List[int]) -> None:
    """Prints summary statistics for a list of token lengths.

    Args:
        name (str): Name of the dataset component being analyzed.
        lengths (List[int]): List of token lengths.
    """
    print(
        f"{name} token lengths: min = {min(lengths)}, "
        f"max = {max(lengths)}, mean = {statistics.mean(lengths):.2f}, "
        f"median = {statistics.median(lengths)}, "
        f"stdev = {statistics.stdev(lengths):.2f}"
    )
    
    q25 = np.percentile(lengths, 25)
    q75 = np.percentile(lengths, 75)
    
    print(
        f"{name} quantiles: 25th = {q25}, "
        f"median = {statistics.median(lengths)}, 75th = {q75}"
    )

def plot_distribution(
    lengths: List[int], title: str, xlabel: str, filename: str
) -> None:
    """Plots and saves a histogram of token lengths.

    Args:
        lengths (List[int]): List of token lengths.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        filename (str): Output file name (not the full path).
    """
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=30, color="skyblue", 
             edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    save_path = IMG_DIR / filename  # Save in docs/assets
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")

def analyze_dataset(data: List[Dict[str, str]], split_name: str) -> None:
    """Analyzes a dataset split and prints validation statistics.

    Args:
        data (List[Dict[str, str]]): The dataset split as a list of 
        dictionaries.
        split_name (str): The name of the dataset split (e.g., "train").
    """
    total: int = len(data)
    label0: int = sum(1 for d in data if d["label"] == "0")
    label1: int = sum(1 for d in data if d["label"] == "1")

    print(f"===== {split_name.upper()} =====")
    print(f"Total examples: {total}")
    print(f"Label 0 (irrelevant): {label0}")
    print(f"Label 1 (relevant): {label1}")
    print("Balanced dataset: " + ("Yes" if label0 == label1 else "No"))

    # Check for repeated questions
    questions: List[str] = [d["question"] for d in data]
    unique_questions: set[str] = set(questions)
    num_duplicates: int = len(questions) - len(unique_questions)

    print(f"Unique questions: {len(unique_questions)}")
    print(f"Repeated questions: {num_duplicates}")

    # Compute token lengths for questions and answers
    question_lengths: List[int] = [get_token_length(q) for q in questions]
    answer_lengths: List[int] = [
        get_token_length(d["answer"]) for d in data
    ]

    print_stats("Question", question_lengths)
    print_stats("Answer", answer_lengths)

    # Compute percentage of answers exceeding 512 tokens
    over_512: int = sum(1 for l in answer_lengths if l > 512)
    print(
        f"Percentage of answers > 512 tokens: "
        f"{over_512 / len(answer_lengths) * 100:.2f}%"
    )

    # Plot distributions (saving in `docs/assets`)
    plot_distribution(
        question_lengths,
        f"{split_name.upper()} - Question Token Length Distribution",
        "Number of tokens",
        f"{split_name}_question_token_distribution.png",
    )

    plot_distribution(
        answer_lengths,
        f"{split_name.upper()} - Answer Token Length Distribution",
        "Number of tokens",
        f"{split_name}_answer_token_distribution.png",
    )

    print("\n")

# --- Process dataset splits ---

SPLITS: List[str] = ["train.csv", "validation.csv", "test.csv"]
DATA_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "data/prepared_data_stratified"
)

for file_name in SPLITS:
    file_path = DATA_DIR / file_name
    if file_path.exists():
        data = load_csv(str(file_path))
        split_name = file_name.split(".")[0]
        analyze_dataset(data, split_name)
    else:
        print(f"⚠️ Warning: {file_path} not found, skipping.")