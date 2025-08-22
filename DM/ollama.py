import os
import sys
import glob
import argparse
import requests
from PyPDF2 import PdfReader

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
def set_public_key_env(public_key):
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    key_line = f"OLLAMA_PUBLIC_KEY={public_key}\n"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
        if any(line.startswith('OLLAMA_PUBLIC_KEY=') for line in lines):
            return  # Key already set
        lines.append(key_line)
        with open(env_path, 'w') as f:
            f.writelines(lines)
    else:
        with open(env_path, 'w') as f:
            f.write(key_line)

OLLAMA_MODEL = "phi3:latest"  # Change to your preferred model
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def read_pdf_text(pdf_path):
    text = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return "\n".join(text)

def collect_pdfs_text(folder):
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    all_text = []
    for pdf in pdf_files:
        print(f"Reading: {pdf}")
        all_text.append(read_pdf_text(pdf))
    return "\n".join(all_text)

def ask_ollama(question, context=""):
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 120 seconds. The model may be processing a complex request."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434."
    except Exception as e:
        return f"Error querying Ollama: {e}"


def embed_texts(texts):
    """Return list of embeddings for the given texts using Ollama embeddings API.
    Falls back to empty list on errors (caller should handle missing embeddings).
    """
    if not texts:
        return []
    try:
        # Ollama embeddings API expects one text at a time; batch for simplicity
        embeddings = []
        for t in texts:
            resp = requests.post(OLLAMA_EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": t}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding") or data.get("embeddings") or data.get("data", [{}])[0].get("embedding")
            if emb is None:
                raise ValueError("No embedding returned")
            embeddings.append(emb)
        return embeddings
    except requests.exceptions.Timeout:
        print(f"Error: Embedding request timed out after 120 seconds")
        return []
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama for embeddings. Please ensure Ollama is running on localhost:11434.")
        return []
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

def main():
    # Set public key in .env if not present
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJaCNy155RCb0TpgmGjEyTdxOqiLT6kCQwI2JOhZEmFi"
    set_public_key_env(public_key)
    parser = argparse.ArgumentParser(description="Ask questions using Ollama and optionally read PDFs.")
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--pdf-folder", help="Folder containing PDFs to use as context.", default=None)
    args = parser.parse_args()

    context = ""
    if args.pdf_folder:
        context = collect_pdfs_text(args.pdf_folder)
        max_context_length = 4000
        if len(context) > max_context_length:
            print(f"Context too long ({len(context)} characters), truncating to {max_context_length} characters.")
            context = context[:max_context_length]
        print(f"Collected context from PDFs ({len(context)} characters).")

    answer = ask_ollama(args.question, context)
    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()