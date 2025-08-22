import glob
import streamlit as st
import os
import requests
import base64
import math
from pathlib import Path
from typing import List
import re

try:
    from rag_store import RagStore
    from ollama import embed_texts
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:latest"
MAX_CHUNK_SIZE = 20000  # Increased for Phi-3's context window
MAX_TOTAL_CHUNKS = 5    # Allow more chunks
TOKEN_ESTIMATE_RATIO = 4 # Rough estimate of characters per token

def chunk_text(text, chunk_size=MAX_CHUNK_SIZE):
    """Split text into larger chunks, respecting Phi-3's context window."""
    # Estimate total tokens (rough approximation)
    estimated_tokens = len(text) / TOKEN_ESTIMATE_RATIO
    
    if estimated_tokens < 100000:  # Safe margin below 123K
        return [text]  # Return full text as single chunk
        
    # If text is too large, split into chunks
    chunks = []
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size:
            if current_chunk:  # Only add if we have content
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    # Return all chunks if under max_total_chunks, otherwise combine some
    if len(chunks) <= MAX_TOTAL_CHUNKS:
        return chunks
    
    # Combine chunks to meet MAX_TOTAL_CHUNKS limit
    chunk_size = math.ceil(len(chunks) / MAX_TOTAL_CHUNKS)
    return [''.join(chunks[i:i + chunk_size]) for i in range(0, len(chunks), chunk_size)]

def list_files(startpath): #Files can be changed 
    tree = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree += f"{indent}📁 {os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree += f"{subindent}📄 {f}\n"
    return tree

def list_pdfs(startpath):
    pdfs = []
    for root, dirs, files in os.walk(startpath):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.abspath(os.path.join(root, f)))
    return pdfs

def get_pdf_text(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# -------- Workflow usage helpers (reads workflow.md without modifying it) --------
_WORKFLOW_USAGE_CACHE = None

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())

def _load_workflow_usage(repo_root: Path):
    global _WORKFLOW_USAGE_CACHE
    if _WORKFLOW_USAGE_CACHE is not None:
        return _WORKFLOW_USAGE_CACHE
    # The file in repo is named with a trailing space: "workflow.md "
    workflow_path = repo_root / "workflow.md "
    usage_map = {}
    doc_to_phases = {}
    if workflow_path.exists():
        text = workflow_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        current_doc = None
        current_info = {}
        # First pass: document sections
        for line in lines:
            m = re.match(r"^###\s+\d+\.\s+(.*\.pdf)\s*$", line)
            if m:
                if current_doc and current_info:
                    usage_map[_sanitize(current_doc)] = current_info
                current_doc = m.group(1).strip()
                current_info = {"focus": None, "key_areas": None, "standards": None}
                continue
            if current_doc:
                if line.strip().lower().startswith("- **focus**:"):
                    current_info["focus"] = line.split(":", 1)[1].strip()
                elif line.strip().lower().startswith("- **key areas**:"):
                    current_info["key_areas"] = line.split(":", 1)[1].strip()
                elif line.strip().lower().startswith("- **standards covered**:"):
                    current_info["standards"] = line.split(":", 1)[1].strip()
        if current_doc and current_info:
            usage_map[_sanitize(current_doc)] = current_info
        # Second pass: phases -> primary documents
        phase = None
        for i, line in enumerate(lines):
            if line.startswith("### Phase "):
                phase = line.replace("### ", "").strip()
            if "**Primary Documents**" in line and phase:
                # Extract comma-separated list after the colon
                raw = line.split(":", 1)[1]
                docs = [d.strip() for d in re.split(r",|;", raw) if d.strip()]
                for d in docs:
                    key = _sanitize(d)
                    doc_to_phases.setdefault(key, set()).add(phase)
    _WORKFLOW_USAGE_CACHE = (usage_map, doc_to_phases)
    return _WORKFLOW_USAGE_CACHE

def get_workflow_usage_for(repo_root: Path, file_path: str):
    usage_map, doc_to_phases = _load_workflow_usage(repo_root)
    base = os.path.basename(file_path)
    s = _sanitize(base)
    # find best match by substring
    best_key = None
    for key in usage_map.keys():
        if key in s or s in key:
            best_key = key
            break
    if not best_key:
        # Try fuzzy by dropping years/spaces
        tokens = re.findall(r"[a-z0-9]+", s)
        for key in usage_map.keys():
            if all(tok in key for tok in tokens[:3]):
                best_key = key
                break
    if not best_key:
        return None
    info = usage_map.get(best_key, {})
    phases = sorted(doc_to_phases.get(best_key, []))
    return {"focus": info.get("focus"), "key_areas": info.get("key_areas"), "standards": info.get("standards"), "phases": phases}

# ----------------------------------------------------------------------------

def ask_ollama(question, context=""):
    """Process Ollama requests with improved error handling."""
    if not context:
        return process_single_request(question)

    chunks = chunk_text(context)
    total_chunks = len(chunks)
    st.info(f"Processing {total_chunks} chunks of context")
    
    # Process each chunk and collect valid responses
    responses = []
    for i, chunk in enumerate(chunks):
        st.info(f"Processing chunk {i+1}/{total_chunks}, length: {len(chunk)}")
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "prompt": f"Context:\n{chunk}\n\nQuestion: {question}\nAnswer based on this context:",
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            chunk_response = response.json().get("response", "").strip()
            if chunk_response:
                responses.append(chunk_response)
                st.info(f"Successfully processed chunk {i+1}/{total_chunks}")
        except Exception as e:
            st.warning(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    # If we got any valid responses, combine them
    if responses:
        try:
            # Create a summary prompt
            summary_prompt = (
                "Based on the following information:\n\n" +
                "\n".join(f"- {r}" for r in responses) +
                f"\n\nProvide a concise answer to the question: {question}"
            )
            
            final_response = requests.post(
                OLLAMA_API_URL,
                json={
                    "prompt": summary_prompt,
                    "stream": False
                }
            )
            final_response.raise_for_status()
            return final_response.json().get("response", "").strip()
        except Exception as e:
            return f"Error in final summary: {str(e)}"
    else:
        return "Could not process context. Please try a shorter document or rephrase your question."

def process_single_request(question):
    """Process a single question without context."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"Question: {question}\nAnswer:",
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error querying Ollama: {e}"

st.title("The Data Management Assistant")

# Resolve docs directory relative to repo root to work locally and in containers
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DOCS_DIR = REPO_ROOT / "it-management-and-audit-source-main"

if BASE_DOCS_DIR.exists():
    tree_str = list_files(str(BASE_DOCS_DIR))
    pdf_files = list_pdfs(str(BASE_DOCS_DIR))
else:
    tree_str = f"Directory not found: {BASE_DOCS_DIR}"
    pdf_files = []

selected_pdf = st.selectbox("Select a PDF to view", ["None"] + pdf_files, help="Choose a PDF file to display , the selected PDF will also be used for context for prompting")

# Show the selected PDF in the main body
if selected_pdf and selected_pdf != "None":
    col_pdf, col_info = st.columns([2, 1], gap="large")
    with col_pdf:
        with open(selected_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    with col_info:
        usage = get_workflow_usage_for(REPO_ROOT, selected_pdf)
        st.subheader("Workflow usage")
        if usage:
            if usage.get("focus"):
                st.write(f"- **Focus**: {usage['focus']}")
            if usage.get("key_areas"):
                st.write(f"- **Key areas**: {usage['key_areas']}")
            if usage.get("standards"):
                st.write(f"- **Standards**: {usage['standards']}")
            if usage.get("phases"):
                st.write(f"- **Used in phases**: {', '.join(usage['phases'])}")
        else:
            st.info("No workflow mapping found for this document.")

question = st.text_input("Enter your question for Ollama:")

use_rag = st.checkbox("Use pgvector retrieval (RAG)", value=False, help="Requires Postgres + pgvector and an Ollama embedding model.")

if st.button("Ask Ollama"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        context = ""
        if use_rag and RAG_AVAILABLE:
            try:
                st.info("🔄 Using RAG (pgvector) path for context retrieval...")
                store = RagStore()
                store.ensure_schema()
                # Ingest selected document if needed
                selected_paths: List[str] = []
                if selected_pdf != "None":
                    selected_paths = [selected_pdf]
                    with st.spinner("Indexing selected PDF if needed..."):
                        from ollama import embed_texts as _emb
                        added, changed, skipped = store.ingest_pdf(selected_pdf, embedder=_emb)
                        st.info(f"Index ready: {added} chunks ({'updated' if changed else 'cached'}), skipped {skipped}")
                # Embed query and search
                q_emb = embed_texts([question])[0]
                results = store.search(q_emb, document_paths=selected_paths, k=6)
                # Build context
                context = "\n\n".join(f"[From {r.document_title}]\n{r.text}" for r in results)
                st.info(f"✅ RAG context built: {len(results)} relevant chunks found, first 100 chars: {context[:100]}")
            except Exception as e:
                st.error(f"RAG path failed, falling back to direct PDF context: {e}")
                context = get_pdf_text(selected_pdf) if selected_pdf != "None" else ""
        else:
            st.info("📄 Using direct PDF text extraction path...")
            if selected_pdf != "None":
                context = get_pdf_text(selected_pdf)
                if context:
                    st.info(f"Successfully loaded PDF content ({len(context)} characters)")

        # Send to Ollama
        answer = ask_ollama(question, context)
        if answer.startswith("Error") or answer.startswith("Could not process"):
            st.error(answer)
        else:
            st.success(answer)


with st.sidebar:
    st.subheader("Source Folder Tree / Knowledge for the agents") 
    st.code(tree_str, language="text")

st.write("The data doctor")