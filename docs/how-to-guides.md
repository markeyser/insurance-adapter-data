# How-To Guides

This section provides step-by-step instructions on **preparing** and **validating** your domain-specific InsuranceQA-based dataset. By following these guides, you can **clean**, **transform**, and **inspect** the dataset before using it to adapt your **Gen AI components**.

---

## 1. Data Preparation

### **1.1 Overview**
This guide walks you through **preparing** the InsuranceQA-derived dataset so it’s **balanced**, **segmented**, and ready for training or fine-tuning. By the end, you’ll have separate **train**, **validation**, and **test** splits in a format suitable for **machine learning pipelines**.

### **1.2 Steps**

1. **Confirm Environment Setup**  
   - Ensure you have **Python 3.10.13** (or a similar version) installed.  
   - Make sure **Poetry** is installed and the virtual environment is active.  
   - Navigate to the project root:

```bash
cd /path/to/your/project
```

2. **Place Raw Data**  
   - The original InsuranceQA files should be placed in:

```
data/raw/
```

   - Ensure the following files are present:

```
data/raw/train
data/raw/dev
data/raw/test1
data/raw/test2
data/raw/vocabulary
data/raw/answers
```

3. **Run the Data Preparation Script**  
   - Execute the script from the project root:

```bash
python src/insuranceqaadapterdata/data_preparation_stratified.py
```

   - **What It Does**:
     - Loads vocabulary and QA data from \`data/raw/\`
     - Cleans and balances the dataset
     - Produces structured **train**, **validation**, and **test** splits
     - Saves the final CSV files in:

```
data/prepared_data_stratified/
```

   - **Expected Output**:

```
data/prepared_data_stratified/train.csv
data/prepared_data_stratified/validation.csv
data/prepared_data_stratified/test.csv
```

### **1.3 Common Troubleshooting**
- **FileNotFoundError**: Ensure the required files exist in \`data/raw/\`.
- **Empty Splits**: If any dataset split is empty, verify the raw files’ structure.

---

## 2. Data Validation

### **2.1 Overview**
Once your dataset is prepared, it’s crucial to validate its structure and quality. This guide explains how to:
- Verify label balance
- Analyze token lengths using a **BERT-like tokenizer**
- Generate **token length distribution plots**

### **2.2 Prerequisites**
- Install dependencies:

```bash
poetry add matplotlib numpy transformers
```

- Ensure **PyTorch**, **TensorFlow**, or **Flax** is installed if needed for tokenizer support.

### **2.3 Steps**

1. **Verify Prepared Data**  
   Ensure the following files exist:

```
data/prepared_data_stratified/train.csv
data/prepared_data_stratified/validation.csv
data/prepared_data_stratified/test.csv
```

2. **Run the Validation Script**

```bash
python src/insuranceqaadapterdata/data_validation.py
```

   - **What It Does**:
     - Loads \`train.csv\`, \`validation.csv\`, and \`test.csv\`
     - Computes statistics (e.g., min/max/mean/median token length)
     - Checks for excessively long answers (>512 tokens)
     - Saves validation plots in:

```
docs/assets/
```

3. **Review Validation Outputs**  
   - **Console Output**: Displays dataset balance and token statistics.
   - **Generated Plots**:
     - Located in \`docs/assets/\`
     - Examples:

```
docs/assets/train_question_token_distribution.png
docs/assets/train_answer_token_distribution.png
docs/assets/validation_question_token_distribution.png
docs/assets/validation_answer_token_distribution.png
docs/assets/test_question_token_distribution.png
docs/assets/test_answer_token_distribution.png
```

   - **Adjustments if Needed**:
     - If many answers exceed **512 tokens**, consider truncation or preprocessing.
     - If label distribution is skewed, check balancing during preparation.

---

## 3. Wrap-Up & Next Steps

Now that your dataset is **prepared** and **validated**, you can:
- **Fine-tune** retrievers and rerankers in a **RAG Q&A system**.
- Use the validated splits for **training or adapting LLMs** for insurance queries.
- **Generate synthetic data** to expand dataset coverage.

For more details, explore additional scripts under:

```
src/insuranceqaadapterdata/
```

---

**Happy Data Prepping & Validating!**