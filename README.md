# Research Paper to Code Agent 🤖

An AI-powered agent that automatically converts ArXiv research papers into runnable PyTorch implementations.

## 🚀 Overview

This project implements an autonomous agent capable of:
1.  **Downloading** research papers from ArXiv given an ID.
2.  **Parsing** the PDF content into structured text using [GROBID](https://github.com/kermitt2/grobid).
3.  **Analyzing** the paper to extract model architecture, hyperparameters, and training details using Large Language Models (LLMs).
4.  **Generating** a complete, runnable PyTorch implementation (`model.py`).

## 📂 Project Structure

The project is organized into a modular MLOps-style structure:

```
AI Agents/
├── src/
│   ├── agent.py                # Main entry point and orchestration
│   ├── config.py               # Configuration (API keys, URLs)
│   ├── tools.py                # Wrappers for ArXiv and GROBID APIs
│   ├── agent_utils.py          # LLM interaction utilities
│   ├── validation_checkers.py  # Code verification logic
│   └── Sub_agents/
│       └── analysis_agent.py   # Specialized agent for paper analysis
├── papers/                     # Directory for downloaded PDFs
├── generated_code/             # Directory for generated Python code
└── notebooks/                  # Exploratory notebooks
```

## 🛠️ Prerequisites

-   **Python 3.9+**
-   **GROBID**:
    -   By default, the agent uses the public demo (`https://kermitt2-grobid.hf.space`).
    -   For better performance/reliability, run it locally using Docker:
        ```bash
        docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
        ```
-   **LLM API Key**:
    -   **OpenAI** (GPT-4) OR **Google Gemini** API key.

## ⚙️ Installation

1.  Clone the repository (if applicable) or navigate to the project folder.
2.  Install the required dependencies:
    ```bash
    pip install torch numpy requests feedparser beautifulsoup4 google-generativeai openai lxml
    ```

## 🔑 Configuration

1.  Open `src/config.py`.
2.  Set your API keys and select your provider:
    ```python
    # src/config.py
    LLM_PROVIDER = "openai"  # or "google"
    OPENAI_API_KEY = "sk-..."
    GOOGLE_API_KEY = "AIza..."
    ```
    *Alternatively, set `OPENAI_API_KEY` or `GOOGLE_API_KEY` as environment variables.*

## 🏃 Usage

Run the main agent script:

```bash
python src/agent.py
```

By default, it is configured to process the U-Net paper (ID: `1505.04597`). You can change the `PAPER_ID` in the `if __name__ == "__main__":` block of `src/agent.py`.

The generated code will be saved to `generated_code/model.py`.

## 🤖 How It Works

1.  **`ResearchAgent`** (in `src/agent.py`) orchestrates the workflow.
2.  **`ArxivTools`** fetches the paper PDF.
3.  **`GrobidTools`** converts the PDF to TEI XML, and extracts sections (Abstract, Method, etc.).
4.  **`AnalysisAgent`** (in `src/Sub_agents/analysis_agent.py`) prompts the LLM to extract a structured JSON specification of the model.
5.  **`ResearchAgent`** then prompts the LLM to generate the PyTorch code based on that specification.
6.  **`ValidationCheckers`** attempts to verify the generated code by running it.
