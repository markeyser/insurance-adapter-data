# InsuranceQA Adapter Data

## Overview

This repository provides a **domain-specific insurance dataset** designed to adapt **key components in Generative AI projects**—such as **dense retrievers**, **generative models**, and **cross-encoder rerankers**—to the **insurance domain**. Since most AI models are trained on **general-domain data**, they often lack the ability to **fully grasp industry-specific terminology** and **insurance-specific nuances** without additional adaptation.

This dataset **bridges that gap** by offering a structured, **balanced**, and **insurance-focused** question-answer dataset. It is particularly useful for **Retrieval-Augmented Generation (RAG)** systems and other advanced **NLP** solutions in real-world **insurance applications**.

> 📖 **Documentation:**  
> - **[Index](https://markeyser.github.io/insurance-adapter-data/)**.
> - **[Explanations](https://markeyser.github.io/insurance-adapter-data/explanation.html):** Learn the **rationale** behind dataset preparation and validation.  
> - **[How-To Guides](https://markeyser.github.io/insurance-adapter-data/how-to-guides.html):** Step-by-step instructions on processing and validating the dataset.  

---

## 📌 Dataset Source: **InsuranceQA**
This dataset is built upon **InsuranceQA**, a publicly available corpus specifically designed for question-answering in the **insurance domain**.  

- **Reference Paper**:  
  Feng, Minwei, et al.  
  _["Applying Deep Learning to Answer Selection: A Study and An Open Task."](https://arxiv.org/abs/1508.01585)_  
  *Empirical Methods in Natural Language Processing (EMNLP), 2015.*

- **Original Repository**:  
  [https://github.com/shuzi/insuranceQA](https://github.com/shuzi/insuranceQA)

> Our modifications **restructure** the original dataset for compatibility with **modern NLP frameworks** and **domain adaptation** tasks in insurance.

---

## 📊 Dataset Summary

The table below summarizes key statistics for the **train, validation, and test splits**.

| Split        | Total Examples | Relevant (Label 1) | Irrelevant (Label 0) | Unique Questions | Min/Max Tokens (Q) | Min/Max Tokens (A) | Answers > 512 Tokens |
|-------------|--------------:|----------------:|----------------:|----------------:|------------------:|------------------:|------------------:|
| **Train**   | 40,324        | 20,162          | 20,162          | 14,797          | 2 / 59           | 4 / 1,102         | 0.37%             |
| **Validation** | 5,040     | 2,520           | 2,520           | 4,042           | 3 / 46           | 7 / 830           | 0.34%             |
| **Test**    | 5,042         | 2,521           | 2,521           | 4,033           | 3 / 40           | 11 / 1,206        | 0.36%             |

> **Notes:**  
> - **Q (Question) Tokens** → Minimum/Maximum token lengths for questions.  
> - **A (Answer) Tokens** → Minimum/Maximum token lengths for answers.  
> - **Long Sequences** → A very small percentage of answers exceed 512 tokens, which is manageable for most LLMs.

---

## 🎯 Possible Use Cases

This dataset is ideal for **adapting Generative AI components** in insurance-specific applications. Below are key **use cases** where this dataset enhances AI performance:

### **1️⃣ Fine-Tuning Adapters for Dense Retrievers or Generative Models**
- **Problem** → General-purpose **retrievers** and **LLMs** do not fully understand **insurance-specific vocabulary**.  
- **Solution** → Fine-tuning on this dataset improves retrieval precision and **LLM-generated responses**.  
- **Application** → Enhances **RAG Q&A systems** for insurance.

### **2️⃣ Fine-Tuning Cross-Encoding Reranker Models**
- **Problem** → Standard **cross-encoders** may misrank insurance-specific results.
- **Solution** → Fine-tuning a **reranker** improves ranking accuracy in **insurance Q&A pipelines**.

### **3️⃣ Synthetic Data Generation**
- **Problem** → Limited real-world data for niche insurance topics.
- **Solution** → Tools like **Microsoft PromptWizard** can use this dataset for **LLM-based automatic annotation**. Additionally, the underlying LLM itself (**DeepSeek‑R1‑Distill‑Qwen‑14B**) can be **fine-tuned** on this dataset to improve the accuracy and quality of its generated labels.
- **Benefit** → Expands dataset coverage without **manual curation**, while also ensuring that the LLM producing the annotations is **optimized for domain-specific reasoning** via fine-tuning.

---

## 🛠 Installation

### ✅ **Prerequisites**
Ensure you have:
- **Python 3.10.13** (recommended via `pyenv`)
- **Poetry** (dependency management)
- **Git** (version control)
- **Docker** (optional, for containerization)

### 📥 **Clone the Repository**

```bash
git clone https://github.com/your-org/insuranceqa-adapter-data.git
cd insuranceqa-adapter-data
```

📌 Setup the Python Environment

```bash
pyenv local 3.10.13
poetry config virtualenvs.create true
poetry env use $(pyenv which python)
poetry install
poetry shell
```

🚀 Usage

1️⃣ Data Preparation

```bash
python src/insuranceqaadapterdata/data_preparation_stratified.py
```

- Input: Raw files must be placed inside `data/raw/`.
- Output: Processed datasets will be saved in `data/prepared_data_stratified/`.

2️⃣ Data Validation & Analysis

```bash
python src/insuranceqaadapterdata/data_validation.py
```

Output:

- Token length statistics
- Histograms saved in `docs/assets/`

For additional guidance, see the How-To Guides.

📂 Project Structure

```plaintext
├── data
│   ├── raw                        # Original InsuranceQA data
│   ├── prepared_data_stratified   # Stratified splits
├── docs
│   ├── assets                     # Images & validation plots
│   ├── explanation.md             # Detailed preparation/validation rationale
│   ├── how-to-guides.md           # Step-by-step task guides
│   └── index.md                   # Documentation home page
├── scripts
│   └── hooks                      # Git hooks for enforcing rules
├── src
│   └── insuranceqaadapterdata     # Main project module
│       ├── data_preparation_stratified.py  # Data processing pipeline
│       └── data_validation.py               # Dataset validation
├── pyproject.toml                 # Poetry dependencies
└── README.md                      # This file
````

🛠 Development

🔍 Linting & Formatting

```bash
poetry run black src tests
poetry run mypy src
```

✅ Testing

`poetry run pytest tests/`

🔄 Pre-commit Hooks

`poetry run pre-commit install`

👥 Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-branch-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to your fork: `git push origin feature-branch-name`
5. Open a pull request.

For guidelines, check [CONTRIBUTING.md](.github/CONTRIBUTING.md).

📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

📢 Acknowledgments

Built upon **InsuranceQA** by Feng et al.

- 🔗 **Reference Paper:** [_Applying Deep Learning to Answer Selection: A Study and An Open Task_ (EMNLP 2015)](https://arxiv.org/abs/1508.01585)
- 📌 **Original Repo:** [GitHub - shuzi/insuranceQA](https://github.com/shuzi/insuranceQA)
