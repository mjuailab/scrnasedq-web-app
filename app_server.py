import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "pipeline_ipynb"))
sys.path.insert(0, str(BASE_DIR / "pipeline_ipynb" / "pipeline"))

from scrnaseq_agent import create_agent, chat, STATE

st.title("scRNA-seq Interactive Analysis Web App")

# 기본 데이터 디렉터리와 서버 환경 변수
DEFAULT_DATA_DIR = Path(os.environ.get("SCRNASEQ_DATA_DIR", BASE_DIR / "data"))
DEFAULT_H5AD_PATH = os.environ.get("SCRNASEQ_H5AD_PATH", "")
if not DEFAULT_H5AD_PATH:
    default_candidate = DEFAULT_DATA_DIR / "final_merged_data0415.h5ad"
    if default_candidate.exists():
        DEFAULT_H5AD_PATH = str(default_candidate)

st.sidebar.header("Configuration")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
model = st.sidebar.selectbox("Model", ["claude-sonnet-4", "gpt-4o", "llama-3.1-70b"])

if st.sidebar.button("Create Agent"):
    if not openai_key and not openrouter_key:
        st.sidebar.error("Please provide at least one API key (OpenAI or OpenRouter).")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        agent = create_agent(
            openrouter_api_key=openrouter_key,
            agent_model=model,
            annotation_model=model,
        )
        st.session_state.agent = agent
        st.sidebar.success("Agent created!")

st.header("Chat with Agent")
message = st.text_input("Enter your message")
if st.button("Send"):
    if "agent" not in st.session_state:
        st.error("Please create an agent first.")
    elif not message.strip():
        st.error("Please enter a message.")
    else:
        response = chat(st.session_state.agent, message)
        st.write("Response:", response)
        if "→" in response:
            for line in response.split("\n"):
                if "→" in line:
                    path = line.split("→")[-1].strip()
                    if os.path.exists(path):
                        st.image(path)

st.header("Quick Actions")
col1, col2, col3 = st.columns(3)

upload_path = st.session_state.get("uploaded_h5ad_path", "")
uploaded_file = st.file_uploader("Upload H5AD file", type=["h5ad"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        upload_path = tmp.name
    st.session_state.uploaded_h5ad_path = upload_path
    st.success(f"Uploaded H5AD file: {Path(upload_path).name}")

with col1:
    file_path_input = st.text_input("H5AD Path", DEFAULT_H5AD_PATH)
    if st.button("Load H5AD"):
        file_path = upload_path or file_path_input
        if file_path and Path(file_path).exists():
            try:
                STATE.adata = sc.read_h5ad(file_path)
                STATE.adata_path = file_path
                st.success("Data loaded!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.error("Please upload or provide a valid H5AD file path.")

with col2:
    if st.button("Preprocess & Cluster"):
        if "agent" in st.session_state:
            chat(st.session_state.agent, "Preprocess and cluster the data")
            st.success("Preprocessing done!")
        else:
            st.error("Please create an agent first.")

with col3:
    if st.button("Plot UMAP"):
        if STATE.adata is not None and "X_umap" in STATE.adata.obsm:
            try:
                sc.pl.umap(STATE.adata, color=["leiden"], show=False)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"Error plotting UMAP: {e}")
        else:
            st.error("No data loaded or UMAP not computed.")

if STATE.adata is not None:
    st.header("Data Info")
    st.write(f"Shape: {STATE.adata.shape}")
    st.write(f"Observations: {list(STATE.adata.obs.columns)}")
    st.write(f"Variables: {list(STATE.adata.var.columns)}")
