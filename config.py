import os

class Config:
    # External Services
    GROBID_URL = os.getenv("GROBID_URL", "https://kermitt2-grobid.hf.space")
    ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=id:{id}&max_results=1"
    
    # LLM Configuration
    # "google" or "openai"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai") 
    
    # API Keys (Load from environment variables for security)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Paths
    OUTPUT_DIR = "generated_code"
    PAPERS_DIR = "papers"

