import re
import json
from agent_utils import LLMUtils

class AnalysisAgent:
    def __init__(self):
        pass

    def analyze_paper(self, sections):
        """Sends paper sections to LLM to extract implementation details."""
        # Construct a prompt with key sections
        context = ""
        for title in ["abstract", "introduction", "method", "model", "architecture", "experiments"]:
            for key in sections:
                if title in key.lower():
                    context += f"--- SECTION: {key} ---\n{sections[key][:4000]}\n\n"
        
        prompt = f"""
        You are an expert AI Researcher and Engineer specializing in reading scientific papers and converting them into precise implementation specifications.

        Your task is to analyze the research paper text below and extract ONLY information explicitly present in the paper. 
        If any detail is missing or unclear, set its value to null rather than guessing.

        --- PAPER CONTENT START ---
        {context}
        --- PAPER CONTENT END ---

        Produce a JSON object with the following keys:

        {{
        "model_name":                string | null,
        "architecture_description":  string | null,     // Describe layers, input/output shapes, connections, blocks, encoders/decoders, attention, etc.
        "components": {{                             
                "encoder":             string | null,
                "backbone":            string | null,
                "decoder":             string | null,
                "loss_modules":        string | null,
                "extra_modules":       string | null      // e.g., regularizers, graph layers, transformers, attention modules, heads
        }},
        "hyperparameters": {{          
                "learning_rate":       float | null,
                "batch_size":          int | null,
                "optimizer":           string | null,
                "epochs":              int | null,
                "weight_decay":        float | null,
                "scheduler":           string | null,
                "augmentation":        string | null
        }},
        "dataset": {{
                "name":                string | null,
                "input_format":        string | null,     // e.g., images 224x224, token sequences, graphs
                "splits":              string | null
        }},
        "training_procedure":        string | null,     // step-wise training loop, tricks (warmup, pretraining, fine-tuning)
        "evaluation_metrics":        string | null,
        "loss_function":             string | null,
        "implementation_notes":      string | null      // details necessary for PyTorch: shapes, initialization, constraints, equations
        }}

        Rules:
        - Do NOT infer or hallucinate missing details.
        - Extract strictly from the paper text.
        - Output MUST be valid JSON.
        - No explanations, no commentary — JSON only.

        """
        
        print("Sending paper to LLM for analysis...")
        response = LLMUtils.query_llm(prompt)
        
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group(0)
            return json.loads(json_str)
        except Exception as e:
            print("Failed to parse LLM response as JSON:", response)
            return None
