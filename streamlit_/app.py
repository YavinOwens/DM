import glob
import streamlit as st
import os
import requests
import base64
import math

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
MAX_CHUNK_SIZE = 20000  # Increased for Phi-3's context window
MAX_TOTAL_CHUNKS = 5    # Allow more chunks
TOKEN_ESTIMATE_RATIO = 4 # Rough estimate of characters per token
MAX_CHUNK_SIZE = 500
MAX_TOTAL_CHUNKS = 3

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
                pdfs.append(os.path.join(root, f))
    return pdfs

def get_pdf_text(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

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
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"Context:\n{chunk}\n\nQuestion: {question}\nAnswer based on this context:",
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            chunk_response = response.json().get("response", "").strip()
            if chunk_response:  # Only add non-empty responses
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
                f"Based on the following information:\n\n" +
                "\n".join(f"- {r}" for r in responses) +
                f"\n\nProvide a concise answer to the question: {question}"
            )
            
            final_response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": summary_prompt,
                    "stream": False
                }
            )
            final_response.raise_for_status()
            return final_response.json().get("response", "").strip()
        except Exception as e:
            return f"Error in final summary: {str(e)}"
    else:
        return "Could not process the context. Please try with a shorter document or rephrase your question."

def process_single_request(question):
    """Process a single question without context."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"Question: {question}\nAnswer:",
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error querying Ollama: {e}"

st.title("The Data Management Assistant")
base_path = "/workspaces/DM/it-management-and-audit-source-main"
tree_str = list_files(base_path)

base_path = "/workspaces/DM/it-management-and-audit-source-main"
pdf_files = list_pdfs(base_path)

selected_pdf = st.selectbox("Select a PDF to view", ["None"] + pdf_files, help="Choose a PDF file to display , the selected PDF will also be used for context for prompting")

# Show the selected PDF in the main body
if selected_pdf and selected_pdf != "None":
    with open(selected_pdf, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

question = st.text_input("Enter your question for Ollama:")

if st.button("Ask Ollama"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        context = ""
        if selected_pdf != "None":
            context = get_pdf_text(selected_pdf)
            if context:
                st.info(f"Successfully loaded PDF content ({len(context)} characters)")
        answer = ask_ollama(question, context)
        if answer.startswith("Error") or answer.startswith("Could not process"):
            st.error(answer)
        else:
            st.success(answer)

with st.sidebar:
    st.subheader("Source Folder Tree / Knowledge for the agents") 
    st.code(tree_str, language="text")

st.write("The data doctor")