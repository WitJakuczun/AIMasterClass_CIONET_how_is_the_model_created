# Technology Stack

## Core Development
- **Language:** Python 3.11+
- **Dependency Management:** `uv` for lightning-fast installs and reproducible environments.

## Machine Learning & NLP
- **Transformers (Hugging Face):** Primary library for RoBERTa, BERT, and Gemma models.
- **PEFT (Parameter-Efficient Fine-Tuning):** For LoRA/QLoRA and other efficient tuning methods.
- **TRL (Transformer Reinforcement Learning):** For RLHF and other advanced training loops.
- **Scikit-learn:** Used for robust data splitting and evaluation metrics.
- **Polars:** Optimized data manipulation library (replacing Pandas) for handling large datasets efficiently.

## Interface & Interaction
- **Typer:** For building the core CLI tool.
- **Rich:** For creating beautiful, color-coded terminal outputs, tables, and progress bars.
- **Loguru:** For structured, readable, and configurable logging.
- **Google Gemini API:** The brain behind the ChatMLOps natural language interface.

## Quality Assurance
- **Pytest:** The standard framework for unit and integration testing.
