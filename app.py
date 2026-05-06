import streamlit as st
import os
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, "/data/project/banana9903/pipeline_ipynb")
sys.path.insert(0, "/data/project/banana9903/pipeline_ipynb/pipeline")
sys.path.insert(0, "/home/banana9903/.local/lib/python3.12/site-packages")  # user packages

import scanpy as sc
from scrnaseq_agent import create_agent, chat, STATE, plot_umap
import matplotlib.pyplot as plt

st.title("scRNA-seq Interactive Analysis Web App")

# Sidebar for configuration
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
            annotation_model=model
        )
        st.session_state.agent = agent
        st.sidebar.success("Agent created!")

# Main chat interface
st.header("Chat with Agent")
message = st.text_input("Enter your message")
if st.button("Send"):
    if 'agent' not in st.session_state:
        st.error("Please create an agent first.")
    elif not message.strip():
        st.error("Please enter a message.")
    else:
        response = chat(st.session_state.agent, message)
        st.write("Response:", response)
        # Check for plot files in response
        if "→" in response:
            lines = response.split("\n")
            for line in lines:
                if "→" in line:
                    path = line.split("→")[-1].strip()
                    if os.path.exists(path):
                        st.image(path)

# Quick actions
st.header("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    file_path_input = st.text_input("H5AD Path", "/data/project/banana9903/pipeline_ipynb/pipeline/final_merged_data0415.h5ad")
    if st.button("Load H5AD"):
        if file_path_input and os.path.exists(file_path_input):
            try:
                STATE.adata = sc.read_h5ad(file_path_input)
                STATE.adata_path = file_path_input
                st.success("Data loaded!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.error("Please provide a valid H5AD file path.")

with col2:
    if st.button("Preprocess & Cluster"):
        if 'agent' in st.session_state:
            chat(st.session_state.agent, "Preprocess and cluster the data")
            st.success("Preprocessing done!")

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

# Display results
if STATE.adata is not None:
    st.header("Data Info")
    st.write(f"Shape: {STATE.adata.shape}")
    st.write(f"Observations: {list(STATE.adata.obs.columns)}")
    st.write(f"Variables: {list(STATE.adata.var.columns)}")
