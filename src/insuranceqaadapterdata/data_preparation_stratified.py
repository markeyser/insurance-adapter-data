"""
Module for processing question-answer data and creating stratified datasets.

This script loads and processes question-answer pairs, balances the dataset,
and performs a stratified split into train, validation, and test sets. The
final data is saved as CSV files for further use in machine learning models.
"""

import csv
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from typing_extensions import TypedDict

# --- Define structured dictionary types ---


class QAEntry(TypedDict):
    """Represents a question-answer dataset entry.

    Attributes:
        question (List[int]): Tokenized question as a list of word indices.
        good (List[int]): Indices of correct answers.
        bad (List[int]): Indices of incorrect answers.
        answers (List[int]): Indices of possible answers in training data.
    """

    question: List[int]
    good: List[int]
    bad: List[int]
    answers: List[int]


class TriadEntry(TypedDict):
    """Represents a processed triad entry for training/testing.

    Attributes:
        question (str): The question text.
        answer (str): The answer text.
        label (int): Label indicating whether the answer is correct (1)
            or incorrect (0).
    """

    question: str
    answer: str
    label: int  # 1 for positive, 0 for negative


# --- Step 1: Define helper functions ---

DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data/raw"


def get_pickle(filename: str) -> Any:
    """Loads a pickle file from the specified data directory.

    Args:
        filename (str): Name of the pickle file.

    Returns:
        Any: The loaded object from the pickle file.
    """
    file_path: Path = DATA_DIR / filename
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Load vocabulary and answers (ensuring correct types)
vocab: Dict[int, str] = get_pickle("vocabulary")
answers: Dict[int, List[int]] = get_pickle("answers")


def translate_sent(token_list: List[int]) -> str:
    """Converts a list of word indices into a readable sentence.

    Args:
        token_list (List[int]): List of word indices.

    Returns:
        str: The reconstructed sentence as a string.
    """
    return " ".join([vocab[word] for word in token_list])


def translate_answer(answer_index: int) -> str:
    """Converts an answer index into a readable text string.

    Args:
        answer_index (int): The index of the answer in the dataset.

    Returns:
        str: The reconstructed answer as a string.
    """
    return translate_sent(answers[answer_index])


# --- Step 2: Load QA splits and produce triads ---

splits: List[str] = ["train", "dev", "test1", "test2"]
triads: List[TriadEntry] = []
all_answer_indices: List[int] = list(answers.keys())

for split in splits:
    print(f"Processing split: {split}")
    try:
        qa_data: List[QAEntry] = get_pickle(split)
    except Exception as e:
        print(f"Error loading {split}: {e}")
        continue

    for item in qa_data:
        question_text: str = translate_sent(item["question"])

        if "good" in item and "bad" in item:
            for good_idx in item["good"]:
                triads.append(
                    {
                        "question": question_text,
                        "answer": translate_answer(good_idx),
                        "label": 1,
                    }
                )
            for bad_idx in item["bad"]:
                triads.append(
                    {
                        "question": question_text,
                        "answer": translate_answer(bad_idx),
                        "label": 0,
                    }
                )

        elif "answers" in item:
            pos_indices: List[int] = item["answers"]
            for ans_idx in pos_indices:
                triads.append(
                    {
                        "question": question_text,
                        "answer": translate_answer(ans_idx),
                        "label": 1,
                    }
                )

            pos_set: set[int] = set(pos_indices)
            negative_candidates: List[int] = [
                idx for idx in all_answer_indices if idx not in pos_set
            ]

            num_negatives: int = len(pos_indices)
            neg_sample: List[int] = (
                random.sample(negative_candidates, num_negatives)
                if len(negative_candidates) >= num_negatives
                else negative_candidates
            )

            for neg_idx in neg_sample:
                triads.append(
                    {
                        "question": question_text,
                        "answer": translate_answer(neg_idx),
                        "label": 0,
                    }
                )
        else:
            print("Warning: QA item does not have expected keys:", item.keys())

print(f"Total triads before balancing: {len(triads)}")

# --- Step 3: Balance the dataset globally ---

positives: List[TriadEntry] = [ex for ex in triads if ex["label"] == 1]
negatives: List[TriadEntry] = [ex for ex in triads if ex["label"] == 0]

print(f"Number of positives: {len(positives)}")
print(f"Number of negatives: {len(negatives)}")

min_count: int = min(len(positives), len(negatives))
print(f"Balancing to {min_count} examples per class.")

random.shuffle(positives)
random.shuffle(negatives)
balanced_positives: List[TriadEntry] = positives[:min_count]
balanced_negatives: List[TriadEntry] = negatives[:min_count]
balanced_triads: List[TriadEntry] = balanced_positives + balanced_negatives
random.shuffle(balanced_triads)

# --- Step 4: Stratified Split ---


def stratified_split(
    positives_list: List[TriadEntry],
    negatives_list: List[TriadEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[TriadEntry], List[TriadEntry], List[TriadEntry]]:
    """Performs a stratified split while maintaining class balance.

    Args:
        positives_list (List[TriadEntry]): List of positive samples.
        negatives_list (List[TriadEntry]): List of negative samples.
        train_ratio (float, optional): Training set ratio. Defaults to 0.8.
        val_ratio (float, optional): Validation set ratio. Defaults to 0.1.

    Returns:
        Tuple[List[TriadEntry], List[TriadEntry], List[TriadEntry]]:
        Train, validation, and test datasets.
    """
    total_pos: int = len(positives_list)
    total_neg: int = len(negatives_list)

    random.shuffle(positives_list)
    random.shuffle(negatives_list)

    train_data: List[TriadEntry] = (
        positives_list[: int(train_ratio * total_pos)]
        + negatives_list[: int(train_ratio * total_neg)]
    )
    val_data: List[TriadEntry] = (
        positives_list[
            int(train_ratio * total_pos) : int(
                (train_ratio + val_ratio) * total_pos
            )
        ]
        + negatives_list[
            int(train_ratio * total_neg) : int(
                (train_ratio + val_ratio) * total_neg
            )
        ]
    )
    test_data: List[TriadEntry] = (
        positives_list[int((train_ratio + val_ratio) * total_pos) :]
        + negatives_list[int((train_ratio + val_ratio) * total_neg) :]
    )

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


# --- Step 5: Save each split to CSV ---
def save_to_csv(data: List[TriadEntry], filename: Path) -> None:
    """Saves processed data to a CSV file.

    Args:
        data (List[TriadEntry]): The processed dataset.
        filename (Path): The file path where the dataset should be saved.
    """
    if not data:
        print(f"Warning: No data to save in {filename}")
        return

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question", "answer", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ex in data:
            writer.writerow(ex)

    print(f"✅ Saved {len(data)} examples to {filename}")

OUTPUT_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "data/prepared_data_stratified"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_data, val_data, test_data = stratified_split(
    balanced_positives, balanced_negatives
)

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")
print(f"Test: {len(test_data)} examples")

# Save CSV files
save_to_csv(train_data, OUTPUT_DIR / "train.csv")
save_to_csv(val_data, OUTPUT_DIR / "validation.csv")
save_to_csv(test_data, OUTPUT_DIR / "test.csv")

