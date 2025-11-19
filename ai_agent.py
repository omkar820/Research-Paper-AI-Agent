# app.py - Streamlit demo for ResearchPaper->AutoImpl skeleton
import streamlit as st
import subprocess, os, tempfile, json
from pathlib import Path
# re-use functions from earlier cells or import them if packaged
# For simplicity, we will inline minimal helpers.

st.set_page_config(page_title="Paper→Code Demo", layout="wide")

st.title("Research Paper → Auto-Implementation (demo skeleton)")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Input")
    arxiv_id = st.text_input("arXiv ID (e.g., 2305.00001)", key="arxiv")
    uploaded_pdf = st.file_uploader("OR upload PDF", type=["pdf"])
    run_btn = st.button("Ingest & Generate")

with col2:
    st.header("Status / Outputs")
    status = st.empty()
    spec_box = st.empty()
    code_box = st.empty()
    test_box = st.empty()

# Inline simple helper functions (reuse from notebook)
import requests, feedparser
ARXIV_API = "http://export.arxiv.org/api/query?search_query=id:{id}&max_results=1"

import arxiv
import os

def download_arxiv_pdf(arxiv_id, download_dir="arxiv_downloads"):
    """
    Downloads an arXiv paper's PDF and returns its path, title, and abstract.

    Args:
        arxiv_id (str): The arXiv ID of the paper (e.g., "2305.00001").
        download_dir (str): Directory to save the PDF.

    Returns:
        tuple: A tuple containing (pdf_path, title, abstract).
               Returns (None, None, None) if the paper is not found or download fails.
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        os.makedirs(download_dir, exist_ok=True)

        # Download the PDF
        pdf_path = paper.download_pdf(dirpath=download_dir)

        return pdf_path, paper.title, paper.summary
    except Exception as e:
        print(f"Error downloading arXiv paper {arxiv_id}: {e}")
        return None, None, None


import requests, time, os

# Change the default URL to the public demo instance
def grobid_process_pdf(pdf_path, grobid_url="https://kermitt2-grobid.hf.space"): 
    """
    Returns TEI XML string from GROBID's processFulltextDocument endpoint.
    """
    endpoint = f"{grobid_url}/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
        resp = requests.post(endpoint, files=files, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError("GROBID failed", resp.status_code, resp.text[:500])
    return resp.text

# tei_xml - this line was a leftover from a previous cell, removed.
from bs4 import BeautifulSoup

def extract_sections_from_tei(tei_xml):
    soup = BeautifulSoup(tei_xml, 'xml')
    sections = {}
    
    # 1. Extract Abstract
    abstract = soup.find("abstract")
    if abstract:
        sections["abstract"] = abstract.get_text(separator="\n", strip=True)
    
    # 2. Extract Body Sections
    # GROBID puts the main content in <text> -> <body> -> <div>
    body = soup.find("body")
    if body:
        for div in body.find_all("div"):
            # The section title is usually in <head>
            head = div.find("head")
            if head:
                section_title = head.get_text(strip=True)
                # Get the text content of the div, excluding the head
                # (We can just get all text and remove the head text, or iterate siblings)
                content = div.get_text(separator="\n", strip=True)
                
                # Clean up: remove the title from the start if it's duplicated
                if content.startswith(section_title):
                    content = content[len(section_title):].strip()
                    
                sections[section_title] = content
            else:
                # Fallback for sections without headers
                sections[f"section_{len(sections)}"] = div.get_text(separator="\n", strip=True)
                
    return sections


import re, json

def heuristic_spec_extractor(sections):
    """
    An improved conservative extractor that fills a JSON spec skeleton from paper sections.
    It aims to extract more details and handle multiple occurrences where appropriate.
    """
    spec = {
        "model_name": None,
        "components": [],  # This is hard to extract reliably with regex, leaving as placeholder
        "loss": [],
        "optimizer": [],
        "dataset": [],
        "training": {
            "learning_rate": [],
            "batch_size": [],
            "epochs": [],
            "gradient_accumulation_steps": [],
        },
        "evaluation": {
            "metrics": [],
            "results": {}, # Placeholder for structured results if regex can capture
        },
        "evidence": {}
    }

    # Helper to add evidence
    def add_evidence(key, match_text, section_name):
        spec["evidence"].setdefault(key, []).append({"text": match_text, "section": section_name})

    # 1) Model name: look for 'we propose' or 'we present'
    for sec_name in ["abstract", "introduction", "method", "methods", "approach", "model", "architecture"]:
        text = sections.get(sec_name, "")
        # More robust regex for model name, allowing for common prefixes/suffixes
        m = re.search(
            r"(?:we (?:propose|present|introduce|develop|design) (?:a|an|new|novel)?\s+)?([A-Za-z0-9\-\_ ]{3,80}(?:(?:model|network|architecture|framework|system|method|algorithm|approach)\b)?)",
            text, re.I
        )
        if m:
            model_name = m.group(1).strip()
            # Filter out generic terms if they are the only match
            if not re.fullmatch(r"(a|an|new|novel)?\s*(model|network|architecture|framework|system|method|algorithm|approach)", model_name, re.I):
                spec["model_name"] = model_name
                add_evidence("model_name", m.group(0), sec_name)
                break # Assume one primary model name

    # Iterate through all sections for other details
    for sec_name, text in sections.items():
        # 2) Dataset heuristics: look for common datasets
        for ds in ["cifar-10", "cifar10", "imagenet", "mnist", "glue", "squad", "coco", "voc", "wikitext", "librispeech", "common crawl", "pile"]:
            if re.search(r"\b" + re.escape(ds) + r"\b", text, re.I):
                if ds not in spec["dataset"]:
                    spec["dataset"].append(ds)
                    add_evidence("dataset", ds, sec_name)

        # 3) Training parameters: learning rate, batch size, epochs, gradient accumulation
        # Learning Rate
        for lr_match in re.finditer(r"(?:learning rate|lr)[s]?\s*[=:\-]?\s*([0-9.eE\-]+)", text, re.I):
            try:
                lr_val = float(lr_match.group(1))
                if lr_val not in spec["training"]["learning_rate"]:
                    spec["training"]["learning_rate"].append(lr_val)
                    add_evidence("learning_rate", lr_match.group(0), sec_name)
            except ValueError:
                pass # Ignore invalid float conversions

        # Batch Size
        for bs_match in re.finditer(r"(?:batch size|bs)[s]?\s*[=:\-]?\s*([0-9]+)", text, re.I):
            try:
                bs_val = int(bs_match.group(1))
                if bs_val not in spec["training"]["batch_size"]:
                    spec["training"]["batch_size"].append(bs_val)
                    add_evidence("batch_size", bs_match.group(0), sec_name)
            except ValueError:
                pass

        # Epochs
        for ep_match in re.finditer(r"([0-9]+)\s*(?:epochs|epoch)", text, re.I):
            try:
                ep_val = int(ep_match.group(1))
                if ep_val not in spec["training"]["epochs"]:
                    spec["training"]["epochs"].append(ep_val)
                    add_evidence("epochs", ep_match.group(0), sec_name)
            except ValueError:
                pass

        # Gradient Accumulation Steps
        for ga_match in re.finditer(r"(?:gradient accumulation steps|accumulate gradients for)\s*([0-9]+)\s*(?:steps|batches)", text, re.I):
            try:
                ga_val = int(ga_match.group(1))
                if ga_val not in spec["training"]["gradient_accumulation_steps"]:
                    spec["training"]["gradient_accumulation_steps"].append(ga_val)
                    add_evidence("gradient_accumulation_steps", ga_match.group(0), sec_name)
            except ValueError:
                pass

        # 4) Loss & Optimizer heuristics
        # Loss Functions
        for loss_name in ["cross-entropy", "mse", "l1 loss", "smooth l1", "kl divergence", "binary cross-entropy"]:
            if loss_name in text.lower():
                if loss_name not in spec["loss"]:
                    spec["loss"].append(loss_name)
                    add_evidence("loss", loss_name, sec_name)

        # Optimizers
        for optimizer_name in ["adamw", "adam", "sgd", "rmsprop", "adagrad"]:
            if optimizer_name in text.lower():
                if optimizer_name not in spec["optimizer"]:
                    spec["optimizer"].append(optimizer_name)
                    add_evidence("optimizer", optimizer_name, sec_name)

        # 5) Evaluation Metrics
        for metric in ["accuracy", "f1-score", "precision", "recall", "bleu", "rouge", "perplexity", "mAP", "iou", "psnr", "ssim"]:
            if re.search(r"\b" + re.escape(metric) + r"\b", text, re.I):
                if metric not in spec["evaluation"]["metrics"]:
                    spec["evaluation"]["metrics"].append(metric)
                    add_evidence("metrics", metric, sec_name)

    # Clean up empty lists/dicts
    for key in ["loss", "optimizer", "dataset"]:
        if not spec[key]:
            spec[key] = None
    for key, val in spec["training"].items():
        if not val:
            spec["training"][key] = None
    if not spec["evaluation"]["metrics"]:
        spec["evaluation"]["metrics"] = None
    if not spec["evaluation"]["results"]:
        spec["evaluation"]["results"] = None
    if not spec["evidence"]:
        spec["evidence"] = None

    return spec

# BASE_REPO_TEMPLATE was not defined, adding a placeholder
BASE_REPO_TEMPLATE = {
    "dataset_hint": "generic",
    "model_family": "cnn",
    "spec": {}
}

def plan_repo_from_spec(spec):
    """
    Very small planner: chooses templates based on dataset & component types.
    """
    plan = BASE_REPO_TEMPLATE.copy()
    # pick dataset template hint
    ds = spec.get("dataset")
    ds_str = ""
    if isinstance(ds, str):
        ds_str = ds
    elif isinstance(ds, list):
        # If dataset is a list, join its string elements for keyword search
        ds_str = " ".join([item for item in ds if isinstance(item, str)])

    if ds_str:
        ds_str_lower = ds_str.lower()
        if "cifar" in ds_str_lower:
            plan['dataset_hint'] = 'cifar'
        elif "mnist" in ds_str_lower:
            plan['dataset_hint'] = 'mnist'
        else:
            plan['dataset_hint'] = 'generic'
    else:
        plan['dataset_hint'] = 'generic'

    # choose model family hint from components or name
    model_name = spec.get("model_name", "")
    if isinstance(model_name, list):
        model_name = " ".join([item for item in model_name if isinstance(item, str)])

    if any(x in model_name.lower() for x in ["transformer","bert","attention"]):
        plan['model_family'] = 'transformer'
    else:
        plan['model_family'] = 'cnn'
    plan['spec']=spec
    return plan


# Template for model.py (CNN) - save as a string and render with format()
MODEL_PY_TEMPLATE = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    \"\"\"Simple CNN family auto-generated from paper spec.
    Paper-evidence: {evidence}
    Input: expected shape {input_shape}
    Output: num_classes={num_classes}
    \"\"\"
    def __init__(self, num_classes={num_classes}, input_channels={input_channels}):
        super(SimpleCNN, self).__init__()
        # conv block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)  # (B, 128, 1, 1)
        x = torch.flatten(x, 1)  # (B, 128)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # quick smoke test
    m = SimpleCNN(num_classes={num_classes}, input_channels={input_channels})
    dummy = torch.randn(2, {input_channels}, {h}, {w})
    out = m(dummy)
    print("output shape:", out.shape)
"""

# def render_model_template(spec, input_shape=(3,32,32), num_classes=10):
#     inp_ch = input_shape[0]
#     h,w = input_shape[1], input_shape[2]
#     evidence = spec.get("evidence", {}).get("model_name", "paper evidence not found")
#     return MODEL_PY_TEMPLATE.format(evidence=evidence, input_shape=input_shape, num_classes=num_classes,
#                                     input_channels=inp_ch, h=h, w=w)

# # Example:
# code = render_model_template(spec, input_shape=(3,32,32), num_classes=10)
# print(code[:400])


if run_btn:
    workdir = tempfile.mkdtemp(prefix="paper2code_")
    status.info(f"Working directory: {workdir}")
    try:
        # 1) fetch/download pdf
        if uploaded_pdf is not None:
            pdf_path = os.path.join(workdir, uploaded_pdf.name)
            with open(pdf_path, "wb") as f: f.write(uploaded_pdf.getbuffer())
            title = uploaded_pdf.name; abstract = ""
        elif arxiv_id.strip():
            status.info("Downloading from arXiv...")
            pdf_path, title, abstract = download_arxiv_pdf_local(arxiv_id.strip(), workdir)
            status.success(f"Downloaded: {os.path.basename(pdf_path)}")
        else:
            st.error("Provide arXiv ID or upload a PDF"); st.stop()

        # 2) call GROBID
        status.info("Calling GROBID (must be running at localhost:8070)...")
        try:
            tei_xml = grobid_process_pdf_local(pdf_path)
            status.success("GROBID OK")
        except Exception as e:
            status.error("GROBID failed: " + str(e))
            # fallback: show a message and try a naive pdfplumber extraction if available
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                combined = "\n\n".join(pages)
            tei_xml = "<text>" + combined[:20000] + "</text>"

        # 3) extract minimal spec
        sections = extract_sections_minimal(tei_xml)
        spec = heuristic_spec_minimal(sections)
        spec_box.json(spec)

        # 4) generate model file (template for demo)
        model_code = MODEL_PY_TEMPLATE
        code_box.code(model_code, language="python")

        # 5) write to disk and run smoke test
        repo_dir = os.path.join(workdir, "repo")
        os.makedirs(repo_dir, exist_ok=True)
        model_path = os.path.join(repo_dir, "model.py")
        with open(model_path, "w") as f: f.write(model_code)
        test_box.info("Running smoke test (python model.py)...")
        res = subprocess.run(["python", model_path], capture_output=True, text=True, timeout=20)
        if res.returncode == 0:
            test_box.success("Smoke test passed\n" + res.stdout)
        else:
            test_box.error("Smoke test failed\n" + res.stdout + "\n" + res.stderr)

        st.markdown(f"Download repository folder: `{repo_dir}` (on server filesystem).")
    except Exception as e:
        status.error("Pipeline failed: " + str(e))
        st.exception(e)
