import os
from dotenv import load_dotenv

load_dotenv()
import json
import re
from config import Config
from tools import ArxivTools, GrobidTools
from agent_utils import LLMUtils
from validation_checkers import ValidationCheckers
from Sub_agents.analysis_agent import AnalysisAgent
from Sub_agents.summary_agent import SummaryAgent
import markdown
from xhtml2pdf import pisa

class ResearchAgent:
    def __init__(self):
        self.analysis_agent = AnalysisAgent()
        self.summary_agent = SummaryAgent()

    def generate_implementation(self, analysis_json, outdir=Config.OUTPUT_DIR):
        """Generates PyTorch code based on the analysis."""
        os.makedirs(outdir, exist_ok=True)
        
        prompt = f"""
                    You are an Elite AI/ML Systems Engineer with expert-level mastery across:
                    - Deep Learning (PyTorch, TensorFlow, JAX)
                    - Machine Learning (sklearn, XGBoost, LightGBM, CatBoost)
                    - Data Engineering (pandas, numpy, polars, pyarrow)
                    - Visualization (matplotlib, seaborn, plotly)
                    - Model Evaluation, Metrics, and Experimentation
                    - Reproducing methodologies from research papers

                    Your job:
                    Given the specification below, generate a **complete, runnable, production-quality Python file** named exactly `model.py`.

                    Your responsibilities:
                    1. **Understand and interpret the provided methodology/specification (research-paper-level if needed).**
                    2. **Design and implement the full code**, even if the description is abstract — fill in missing details logically.
                    3. **Choose the right tools automatically** (deep learning, classical ML, preprocessing, training loops, evaluation, etc.).

                    Strict Requirements:
                    - The final output MUST be valid standalone Python code (no markdown formatting).
                    - All imports must be included.
                    - All classes and functions must be fully defined.
                    - The file must be executable without modification.
                    - When a model is needed, automatically choose the correct architecture (MLP, CNN, transformer, RNN, GNN, classical ML model, etc.) based on the description.
                    - When preprocessing is needed, implement it fully.
                    - When evaluation is needed, implement metrics, plots, etc.

                    Mandatory Structure Inside `model.py`:
                    1. A top-level `build_model()` or equivalent function/class implementing the core method.
                    2. A `run_demo()` or equivalent function that:
                    - Creates mock/sample data appropriate for the model.
                    - Runs preprocessing (if applicable).
                    - Trains the model briefly (if applicable).
                    - Performs inference.
                    - Prints outputs (shapes + sample predictions).
                    3. A `if __name__ == "__main__":` block calling the demo.
                    4. The script must NEVER rely on external data unless provided in the specification.
                    5. Use any library logically required (torch, sklearn, numpy, pandas, matplotlib, seaborn, statsmodels, scipy, etc.).

                    General Rules:
                    - Code must be clean, modular, and readable.
                    - If the specification describes a pipeline, implement the entire pipeline end-to-end.
                    - If math or algorithms are vaguely described, infer the most scientifically correct interpretation.
                    - If the method implies a custom training loop, implement it.
                    - If the method implies hyperparameters, choose sensible defaults.
                    - If a dataset structure is implied, simulate it.

                    Here is the specification you must implement:
                    {json.dumps(analysis_json, indent=2)}
                    """

        
        print("Generating code...")
        code_response = LLMUtils.query_llm(prompt)
        
        # Extract only the code block, assuming it's enclosed in markdown fences
        # and there might be extraneous text before or after.
        match = re.search(r"```python\s*(.*?)\s*```", code_response, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # If no markdown fences are found, assume the entire response is code
            # and strip potential leading/trailing whitespace/comments.
            code = code_response.strip()
            # Attempt to remove common LLM-generated introductory/concluding remarks
            if code.startswith("```python"):
                code = code.replace("```python", "", 1).strip()
            if code.endswith("```"):
                code = code[:-3].strip()

        filepath = os.path.join(outdir, "model.py")
        with open(filepath, "w") as f:
            f.write(code)
        print(f"Code saved to {filepath}")
        return filepath

    def run(self, paper_id):
        print(f"Starting Research Agent for Paper ID: {paper_id}")
        
        # 1. Download
        pdf_path, title, abstract = ArxivTools.download_paper(paper_id)
        print(f"Paper Title: {title}")
        
        # 2. Parse
        tei_xml = GrobidTools.process_pdf(pdf_path)
        sections = GrobidTools.extract_sections(tei_xml)
        print(f"Extracted {len(sections)} sections.")
        
        # 3. Analyze
        analysis = self.analysis_agent.analyze_paper(sections)
        if not analysis:
            print("Analysis failed.")
            return
            
        print("Analysis Result:", json.dumps(analysis, indent=2))

        # 3.5 Generate Summary
        summary = self.summary_agent.generate_summary(sections)
        if summary:
            summary_path = os.path.join(Config.OUTPUT_DIR, "summary.pdf")
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            
            # Convert Markdown to HTML
            html_text = markdown.markdown(summary)
            
            # Convert HTML to PDF
            with open(summary_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_text, dest=pdf_file)
            
            if not pisa_status.err:
                print(f"Summary saved to {summary_path}")
            else:
                print(f"Failed to generate PDF summary")


        
        # 4. Generate
        code_path = self.generate_implementation(analysis)
        
        # 5. Verify
        if code_path:
            print("\nVerifying generated code...")
            ValidationCheckers.verify_python_file(code_path)

if __name__ == "__main__":
    # Example Usage
    # Ensure API keys are set in environment or Config
    agent = ResearchAgent()
    agent.run("2212.03273")
