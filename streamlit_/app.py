import streamlit as st
import os

import base64

def list_files(startpath): 
    tree = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree += f"{indent}📁 {os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree += f"{subindent}📄 {f}\n"
    return tree

st.title("The Data Management Assistant")
base_path = "/workspaces/DM/it-management-and-audit-source-main"
tree_str = list_files(base_path)


def list_pdfs(startpath):
    pdfs = []
    for root, dirs, files in os.walk(startpath):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, f))
    return pdfs

base_path = "/workspaces/DM/it-management-and-audit-source-main"
pdf_files = list_pdfs(base_path)

selected_pdf = st.selectbox("Select a PDF to view", ["None"] + pdf_files, help="Choose a PDF file to display , the selected PDF will also be used for context for prompting")

if selected_pdf:
    with open(selected_pdf, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.subheader("Source Folder Tree / Knowledge for the agents") 
    st.code(tree_str, language="text")

st.write("The data doctor")

