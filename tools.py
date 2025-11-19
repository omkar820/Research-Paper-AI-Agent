import os
import requests
import feedparser
from bs4 import BeautifulSoup
from config import Config

class ArxivTools:
    @staticmethod
    def download_paper(arxiv_id, outdir=Config.PAPERS_DIR):
        """Downloads a paper from ArXiv given its ID."""
        os.makedirs(outdir, exist_ok=True)
        print(f"Fetching metadata for {arxiv_id}...")
        
        # 1. Get Metadata
        q = Config.ARXIV_API_URL.format(id=arxiv_id)
        resp = requests.get(q, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        if not feed.entries:
            raise ValueError("arXiv id not found.")
        entry = feed.entries[0]
        title = entry.get('title', 'Unknown Title')
        abstract = entry.get('summary', '')
        
        # 2. Construct PDF URL
        pdf_url = None
        for link in entry.get('links', []):
            if link.get('rel') == 'alternate' and 'arxiv.org/abs' in link.get('href', ''):
                pdf_url = link['href'].replace('/abs/','/pdf/') + ".pdf"
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
        # 3. Download PDF
        pdf_path = os.path.join(outdir, f"{arxiv_id}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Downloading PDF from {pdf_url}...")
            r = requests.get(pdf_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(1024*16):
                    if chunk: f.write(chunk)
            print(f"Saved to {pdf_path}")
        else:
            print(f"PDF already exists at {pdf_path}")
            
        return pdf_path, title, abstract

class GrobidTools:
    @staticmethod
    def process_pdf(pdf_path):
        """Uploads PDF to GROBID and returns TEI XML."""
        endpoint = f"{Config.GROBID_URL}/api/processFulltextDocument"
        print(f"Processing {pdf_path} with GROBID at {Config.GROBID_URL}...")
        
        with open(pdf_path, "rb") as f:
            files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
            try:
                resp = requests.post(endpoint, files=files, timeout=300)
                if resp.status_code != 200:
                    raise RuntimeError(f"GROBID error: {resp.status_code} {resp.text[:200]}")
                return resp.text
            except requests.exceptions.ConnectionError:
                raise RuntimeError(f"Could not connect to GROBID at {Config.GROBID_URL}. Is it running?")

    @staticmethod
    def extract_sections(tei_xml):
        """Parses TEI XML to extract text sections organized by header."""
        soup = BeautifulSoup(tei_xml, 'xml')
        sections = {}
        
        # Abstract
        abstract = soup.find("abstract")
        if abstract:
            sections["abstract"] = abstract.get_text(separator="\n", strip=True)
        
        # Body
        body = soup.find("body")
        if body:
            for div in body.find_all("div"):
                head = div.find("head")
                if head:
                    title = head.get_text(strip=True)
                    content = div.get_text(separator="\n", strip=True)
                    # Remove title from content if present
                    if content.startswith(title):
                        content = content[len(title):].strip()
                    sections[title] = content
                else:
                    # Fallback for untitled sections
                    sections[f"section_{len(sections)}"] = div.get_text(separator="\n", strip=True)
        return sections
