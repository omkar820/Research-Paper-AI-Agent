import google.generativeai as genai
import openai
from config import Config

class LLMUtils:
    @staticmethod
    def query_llm(prompt):
        """Wrapper to call the selected LLM provider."""
        if Config.LLM_PROVIDER == "google":
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set in Config or Environment.")
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
            
        elif Config.LLM_PROVIDER == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in Config or Environment.")
            openai.api_key = Config.OPENAI_API_KEY
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        return ""
