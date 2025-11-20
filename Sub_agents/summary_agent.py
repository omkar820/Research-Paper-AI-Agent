from agent_utils import LLMUtils

class SummaryAgent:
    def __init__(self):
        pass

    def generate_summary(self, sections):
        """
        Generates a comprehensive summary of the research paper tailored for an AI Engineer.
        """
        # Construct context from key sections
        context = ""
        # Prioritize sections that are most relevant for a summary
        target_sections = ["abstract", "introduction", "method", "model", "architecture", "experiments", "conclusion", "results"]
        
        for title in target_sections:
            for key in sections:
                if title in key.lower():
                    # Limit section length to avoid context window issues, but keep enough for summary
                    context += f"--- SECTION: {key} ---\n{sections[key][:5000]}\n\n"

        prompt = f"""
        You are an expert AI Researcher and Engineer. Your task is to write a comprehensive, technical summary of the following research paper for another AI Engineer.

        The summary should be detailed but concise, focusing on the "how" and "why".

        --- PAPER CONTENT START ---
        {context}
        --- PAPER CONTENT END ---

        Please structure the summary as follows in Markdown format:

        # [Paper Title] - Summary

        ## 1. Problem Statement
        - What problem is this paper trying to solve?
        - Why is it important?
        - What are the limitations of existing approaches?

        ## 2. Key Contributions
        - What are the main novelties proposed? (e.g., new architecture, loss function, training method, dataset)

        ## 3. Methodology / Architecture
        - Explain the core technical approach in detail.
        - Describe the model architecture, key components, and how they interact.
        - Mention any specific mathematical formulations or algorithms if critical.

        ## 4. Experiments & Results
        - What datasets were used?
        - What were the main results? (Quantitative metrics)
        - How does it compare to SOTA?

        ## 5. Implementation Details for Engineers
        - Key hyperparameters (learning rate, batch size, optimizer, etc.).
        - Hardware requirements or computational complexity.
        - Any specific tricks or constraints mentioned (e.g., initialization, data augmentation).

        ## 6. Conclusion & Impact
        - Brief wrap-up of the paper's significance.

        Output strictly the Markdown summary.
        """

        print("Generating paper summary...")
        summary = LLMUtils.query_llm(prompt)
        return summary
