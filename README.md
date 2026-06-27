# GenAI QA Evaluation Framework

A comprehensive evaluation framework for Generative AI applications, leveraging [DeepEval](https://github.com/confident-ai/deepeval) and Google's Gemini models as judges. This project provides an automated, parameterized test suite using `pytest` to evaluate LLM outputs across various dimensions including correctness, RAG quality, and security.

## Features

*   **RAG Evaluation (`tests/test_rag/`)**: Assesses retrieval-augmented generation pipelines using metrics like Contextual Precision, Contextual Recall, Contextual Relevancy, Answer Relevancy, and Faithfulness.
*   **Security & Safety Testing (`tests/test_security/`)**: Evaluates model outputs for Bias, Toxicity, PII Leakage, Misuse, and adherence to Non-Advice guidelines.
*   **LLM Behavioral Evaluation (`tests/test_llm/`)**: Tests core behaviors including Correctness, Hallucination, and Professionalism (including multi-turn conversational evaluation).
*   **Robust Test Infrastructure**:
    *   Data-driven testing with dataset parameterization (`datasets/testdata.json` & `conversations.json`).
    *   Built-in retry mechanisms for API rate limits ("429 Too Many Requests").
    *   Centralized configuration via `.env` and `pytest.ini`.

## Setup and Installation

1.  **Clone the repository** (if you haven't already).
2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables**:
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Update the `.env` file with your Google API Key and preferred model:
        ```env
        GOOGLE_API_KEY=your_google_api_key_here
        MODEL=gemini-1.5-pro
        ```

## Running Tests

The test suite is powered by `pytest`. You can run the entire suite or target specific categories.

Run all tests:
```bash
pytest
```

Run specific test categories using markers (defined in `pytest.ini`):
```bash
pytest -m rag       # Run only RAG evaluations
pytest -m safety    # Run only Security and Safety evaluations
```

Run tests in a specific directory:
```bash
pytest tests/test_security/
```

## Project Structure

*   `datasets/`: JSON files containing evaluation scenarios and conversations.
*   `tests/`: The `pytest` test suite, organized by category (`test_llm`, `test_rag`, `test_security`).
*   `utils/`: Shared utilities, including dataset loaders and judge setup.
*   `reports/` & `logs/`: Directory for test execution logs and evaluation reports.