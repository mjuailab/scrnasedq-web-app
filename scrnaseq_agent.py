"""
scRNA-seq Interactive Analysis Agent
=====================================
LangChain 1.2.x / LangGraph-based agent for interactive single-cell RNA-seq analysis.
Supports OpenRouter for model switching (GPT-4o, Claude, Llama, DeepSeek, Gemini, etc.).

Usage (notebook):
    from scrnaseq_agent import create_agent, chat, run_interactive_session
    agent = create_agent(openai_api_key="sk-...", openrouter_api_key="sk-or-...")
    chat(agent, "Load /path/to/data.h5ad and annotate cell types")

Usage (CLI):
    ~/.conda/envs/ADPN/bin/python scrnaseq_agent.py --openai-key sk-...
"""

import os
import sys
import warnings
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings("ignore")

# Add pipeline directory to path so local modules are importable
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "pipeline"))  # preprocessing.py etc. live here

# LangChain 1.2.x / LangGraph imports
from langchain.tools import tool
from langchain.agents import create_agent as _lc_create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==============================================================================
# OpenRouter model catalogue
# ==============================================================================

OPENROUTER_MODELS: Dict[str, str] = {
    "gpt-4o":             "openai/gpt-4o",
    "gpt-4o-mini":        "openai/gpt-4o-mini",
    "gpt-4-turbo":        "openai/gpt-4-turbo",
    "gpt-3.5-turbo":      "openai/gpt-3.5-turbo",
    "gpt-5.4pro" :        "openai/gpt-5.4-pro",
    "claude-3.5-sonnet":  "anthropic/claude-3.5-sonnet",
    "claude-3-opus":      "anthropic/claude-3-opus",
    "claude-3-haiku":     "anthropic/claude-3-haiku",
    "claude-sonnet-4":    "anthropic/claude-sonnet-4",
    "claude-opus-4":      "anthropic/claude-opus-4",
    "llama-3.1-70b":      "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b":       "meta-llama/llama-3.1-8b-instruct",
    "llama-3.3-70b":      "meta-llama/llama-3.3-70b-instruct",
    "llama-4-maverick":   "meta-llama/llama-4-maverick",
    "llama-4-scout":      "meta-llama/llama-4-scout",
    "gemini-pro-1.5":     "google/gemini-pro-1.5",
    "gemini-flash-1.5":   "google/gemini-flash-1.5",
    "gemini-2.0-flash":   "google/gemini-2.0-flash-001",
    "mistral-large":      "mistralai/mistral-large",
    "mixtral-8x7b":       "mistralai/mixtral-8x7b-instruct",
    "deepseek-v3":        "deepseek/deepseek-chat",
    "deepseek-r1":        "deepseek/deepseek-r1",
    "qwen-72b":           "qwen/qwen-2.5-72b-instruct",
}

# ==============================================================================
# Session State (module-level singleton)
# ==============================================================================

class _ScRNAState:
    def __init__(self):
        self.adata              = None
        self.adata_path         = None
        self.cell_type_adatas   = {}
        self.results            = {"markers": {}, "deg": {}, "enrichment": {}}
        self.tissue_type        = "skin"
        self.model_name         = "gpt-4o"
        self.use_openrouter     = False
        self.openai_api_key     = os.environ.get("OPENAI_API_KEY", "")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.output_dir         = Path("./scrnaseq_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def annotation_api_key(self):
        if self.use_openrouter and self.openrouter_api_key:
            return self.openrouter_api_key
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    def annotation_model_id(self):
        if self.use_openrouter:
            return OPENROUTER_MODELS.get(self.model_name, self.model_name)
        return self.model_name


STATE = _ScRNAState()

# ==============================================================================
# Helpers
# ==============================================================================

def _save_fig(filename):
    p = STATE.output_dir / filename
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close("all")
    return p


def _adata_info(adata):
    lines = [
        f"  Cells : {adata.n_obs:,}",
        f"  Genes : {adata.n_vars:,}",
        f"  obs   : {list(adata.obs.columns)}",
        f"  obsm  : {list(adata.obsm.keys())}",
    ]
    if "LLM_annotation" in adata.obs.columns:
        vc = adata.obs["LLM_annotation"].value_counts()
        lines.append("  Cell types:")
        for ct, n in vc.items():
            lines.append(f"    {ct}: {n:,} ({100*n/adata.n_obs:.1f}%)")
    return "\n".join(lines)


def _rank_genes_df(adata, groupby):
    """Run Wilcoxon rank_genes_groups and return a DataFrame with column names
    that match what annotation_pipeline.filter_marker_genes() expects:
      cluster | gene | avg_log2FC | p_val_adj | pct.1 | pct.2
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon",
                            pts=True, use_raw=False)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    # Normalise to annotation_pipeline column convention
    df = df.rename(columns={
        "group":           "cluster",
        "names":           "gene",
        "logfoldchanges":  "avg_log2FC",
        "pvals_adj":       "p_val_adj",
        "pct_nz_group":    "pct.1",
        "pct_nz_reference":"pct.2",
    })
    return df



def _autosave(label: str = "checkpoint") -> Path:
    """Save STATE.adata to output_dir with a step label.
    Called automatically after each major analysis step.
    """
    if STATE.adata is None:
        return None
    path = STATE.output_dir / f"adata_{label}.h5ad"
    STATE.adata.write_h5ad(path)
    return path

def _run_llm_annotation(adata, tissue_type: str, groupby: str = "leiden") -> object:
    """Run LLM cell type annotation using the configured model (supports OpenRouter).

    Handles:
    - Ensuring 'cluster' obs column exists (copies from groupby if needed)
    - Calling the LLM via ChatOpenAI (supports base_url for OpenRouter)
    - Storing results in adata.obs['LLM_annotation']
    """
    import json
    from tqdm import tqdm
    from annotation_pipeline import filter_marker_genes

    # 1. Ensure 'cluster' column exists (annotation_pipeline requirement)
    if "cluster" not in adata.obs.columns:
        if groupby in adata.obs.columns:
            adata.obs["cluster"] = adata.obs[groupby].astype(str)
        else:
            raise ValueError(
                f"Neither 'cluster' nor '{groupby}' found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

    # 2. Filter markers (modifies adata in-place — do NOT assign return value)
    filter_marker_genes(adata)
    marker_df = adata.uns["marker_list"]

    # 3. Build LLM client via LangChain (handles OpenRouter base_url transparently)
    llm = _make_llm(
        model_name=STATE.model_name,
        openai_api_key=STATE.openai_api_key,
        openrouter_api_key=STATE.openrouter_api_key,
        use_openrouter=STATE.use_openrouter,
        temperature=0.0,
    )

    # 4. Annotate each cluster
    cluster_annotations = {}
    unique_clusters = marker_df["cluster"].unique()
    print(f"Annotating {len(unique_clusters)} clusters with {STATE.model_name}...")

    for cluster in tqdm(unique_clusters, desc="LLM annotation"):
        genes = marker_df[marker_df["cluster"] == cluster]["gene"].tolist()
        gene_str = ", ".join(genes[:50])  # cap to avoid token overflow

        prompt = (
            f"Single-cell RNA-seq cluster marker genes: {gene_str}\n\n"
            f"Tissue: {tissue_type} (skin disease study: atopic dermatitis / prurigo nodularis).\n"
            "Identify the cell type. Return ONLY a single-line JSON: "
            '{"cell_type": "Cell type name"}'
        )
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            cell_type = data.get("cell_type", "Unknown")
        except Exception as e:
            print(f"  Cluster {cluster}: annotation failed ({e}), marking Unknown")
            cell_type = "Unknown"

        cluster_annotations[str(cluster)] = cell_type
        print(f"  Cluster {cluster}: {cell_type}")

    # 5. Map back to obs
    adata.obs["LLM_annotation"] = (
        adata.obs["cluster"].astype(str).map(cluster_annotations)
    )

    # 6. Save JSON
    out_dir = STATE.output_dir / "LLM_res"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "llm_cluster_annotations.json", "w") as f:
        json.dump(cluster_annotations, f, indent=4)

    print(f"\nAnnotation complete. Results saved to {out_dir}")
    return adata

# ==============================================================================
# Tools
# ==============================================================================

@tool
def load_h5ad(file_path: str) -> str:
    """Load an .h5ad file to begin the analysis session.

    Args:
        file_path: Path to the .h5ad file.
    """
    try:
        STATE.adata = sc.read_h5ad(file_path)
        STATE.adata_path = file_path
        return f"Loaded: {file_path}\n" + _adata_info(STATE.adata)
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def get_session_state() -> str:
    """Return a summary of the current analysis session."""
    if STATE.adata is None:
        return "No data loaded. Use load_h5ad() to begin."
    return "\n".join([
        "=== Analysis Session ===",
        f"File       : {STATE.adata_path}",
        f"Model      : {STATE.model_name} ({'OpenRouter' if STATE.use_openrouter else 'OpenAI direct'})",
        f"Output dir : {STATE.output_dir}",
        "",
        _adata_info(STATE.adata),
        "",
        f"Cell-type sub-objects : {list(STATE.cell_type_adatas.keys())}",
        f"DEG results           : {list(STATE.results['deg'].keys())}",
        f"Enrichment results    : {list(STATE.results['enrichment'].keys())}",
    ])


@tool
def preprocess_and_cluster(
    batch_key: str = "batch",
    resolution: float = 0.8,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
) -> str:
    """Preprocess and cluster the AnnData.

    If the data already has a scVI latent space (X_scVI) or Harmony embedding
    (X_pca_harmony), only Leiden clustering + UMAP are run (fast path).
    Otherwise full preprocessing is performed: HVG → normalisation → PCA →
    Harmony batch correction → Leiden → UMAP.

    Args:
        batch_key: obs column for Harmony batch correction (default 'batch').
        resolution: Leiden clustering resolution (default 0.8).
        n_top_genes: Number of highly variable genes for full preprocessing (default 2000).
        n_pcs: PCA components for full preprocessing (default 30).
    """
    if STATE.adata is None:
        return "No data loaded. Call load_h5ad() first."
    try:
        adata = STATE.adata

        # ── Fast path: embedding already exists, just cluster ──────────────────
        embed_key = None
        for k in ["X_scVI", "X_pca_harmony", "X_pca"]:
            if k in adata.obsm:
                embed_key = k
                break

        if embed_key and "leiden" not in adata.obs.columns:
            print(f"Detected existing embedding '{embed_key}' — running Leiden + UMAP only.")
            sc.pp.neighbors(adata, use_rep=embed_key)
            sc.tl.leiden(adata, resolution=resolution)
            if "X_umap" not in adata.obsm:
                sc.tl.umap(adata)
            mode = f"fast (used {embed_key})"

        elif "leiden" in adata.obs.columns:
            mode = "skipped (leiden already exists)"

        else:
            # ── Full preprocessing ─────────────────────────────────────────────
            from preprocessing import preprocessing_subcluster
            adata = preprocessing_subcluster(
                adata, batch_key=batch_key, n_pcs=n_pcs,
                resolution=resolution, hvg_top_genes=n_top_genes,
            )
            mode = "full (HVG → Harmony → Leiden → UMAP)"

        STATE.adata = adata
        out = STATE.output_dir / "preprocessed.h5ad"
        adata.write_h5ad(out)

        n_clusters = adata.obs["leiden"].nunique()
        sc.pl.umap(adata, color=["leiden"], show=False)
        _save_fig("umap_leiden.png")

        saved = _autosave("01_clustered")
        return (
            f"Clustering complete [{mode}]\n"
            f"  Leiden clusters : {n_clusters}\n"
            f"  UMAP computed   : {'X_umap' in adata.obsm}\n"
            f"  Auto-saved      : {saved}\n\n"
            f"Cluster sizes:\n{adata.obs['leiden'].value_counts().sort_index().to_string()}"
        )
    except Exception as e:
        return f"TOOL ERROR in preprocess_and_cluster:\n{e}\n{traceback.format_exc()}"


@tool
def compute_marker_genes(groupby: str = "leiden", top_n: int = 100) -> str:
    """Compute marker genes per group using Wilcoxon rank-sum test.

    If groupby='leiden' but that column does not yet exist, Leiden clustering
    is run automatically first using any available embedding.

    Args:
        groupby: obs column to group by ('leiden' or 'LLM_annotation').
        top_n: Max marker genes kept per group after significance filtering.
    """
    if STATE.adata is None:
        return "No data loaded."
    try:
        adata = STATE.adata

        # Auto-run clustering when leiden is requested but absent
        if groupby == "leiden" and "leiden" not in adata.obs.columns:
            print("leiden column not found — running clustering first...")
            result = preprocess_and_cluster.invoke({})
            print(result)
            adata = STATE.adata  # refresh after clustering

        if groupby not in adata.obs.columns:
            avail = [c for c in adata.obs.columns
                     if adata.obs[c].dtype.name in ("category", "object")]
            return (
                f"TOOL ERROR: column '{groupby}' not found in adata.obs.\n"
                f"Available categorical columns: {avail}"
            )

        df = _rank_genes_df(adata, groupby)
        sig = (
            df[(df["p_val_adj"] < 0.05) & (df["avg_log2FC"].abs() > 0.5)]
            .sort_values("avg_log2FC", ascending=False)
            .groupby("cluster").head(top_n)
            .reset_index(drop=True)
        )
        adata.uns["find_markers"] = df
        STATE.results["markers"][groupby] = sig
        out = STATE.output_dir / f"markers_{groupby}.csv"
        sig.to_csv(out, index=False)
        saved = _autosave("02_with_markers")
        lines = [f"Marker genes for '{groupby}' ({sig['cluster'].nunique()} groups) → {out}",
                 f"Auto-saved adata: {saved}"]
        for grp in sorted(sig["cluster"].unique()):
            sub = sig[sig["cluster"] == grp]
            top = sub.nlargest(5, "avg_log2FC")["gene"].tolist()
            lines.append(f"  {grp}: {sub.shape[0]} sig genes | top: {', '.join(top)}")
        return "\n".join(lines)
    except Exception as e:
        return f"TOOL ERROR in compute_marker_genes:\n{e}\n{traceback.format_exc()}"


@tool
def annotate_cell_types(tissue_type: str = "skin") -> str:
    """Annotate clusters with cell types using an LLM and marker genes.
    Works with OpenAI and OpenRouter (Claude, Llama, DeepSeek, etc.).

    Args:
        tissue_type: Tissue context for LLM (e.g. 'skin', 'pbmc', 'lung').
    """
    if STATE.adata is None:
        return "No data loaded."
    if not STATE.annotation_api_key():
        return "No API key. Call switch_model() with your key."
    try:
        STATE.tissue_type = tissue_type

        # Ensure marker genes are computed
        groupby = "leiden" if "leiden" in STATE.adata.obs.columns else "cluster"
        if "find_markers" not in STATE.adata.uns:
            result = compute_marker_genes.invoke({"groupby": groupby})
            print(result)

        adata = _run_llm_annotation(STATE.adata, tissue_type, groupby=groupby)
        STATE.adata = adata

        vc = adata.obs["LLM_annotation"].value_counts()
        sc.pl.umap(adata, color=["LLM_annotation"], show=False)
        fig_path = _save_fig("umap_annotation.png")
        saved = _autosave("03_annotated")
        lines = [f"Annotation complete → UMAP: {fig_path}", f"Auto-saved adata: {saved}"]
        for ct, n in vc.items():
            lines.append(f"  {ct}: {n:,} ({100*n/adata.n_obs:.1f}%)")
        return "\n".join(lines)
    except Exception as e:
        return f"TOOL ERROR in annotate_cell_types:\n{e}\n{traceback.format_exc()}"


@tool
def annotate_subtypes(
    cell_type: str,
    resolution: float = 0.5,
    n_top_genes: int = 1500,
) -> str:
    """Subset one cell type and annotate its subtypes with LLM
    (e.g. Keratinocyte -> Basal, Spinous, Granular).

    Args:
        cell_type: Cell type label to subset (partial match on LLM_annotation).
        resolution: Leiden resolution for sub-clustering (default 0.5).
        n_top_genes: HVG count for sub-clustering (default 1500).
    """
    if STATE.adata is None:
        return "No data loaded."
    if "LLM_annotation" not in STATE.adata.obs.columns:
        return "Run annotate_cell_types() first."
    try:
        from preprocessing import preprocessing_subcluster
        mask = STATE.adata.obs["LLM_annotation"].str.contains(
            cell_type, case=False, na=False)
        if mask.sum() == 0:
            avail = STATE.adata.obs["LLM_annotation"].unique().tolist()
            return f"No cells matched '{cell_type}'. Available: {avail}"
        adata_sub = STATE.adata[mask].copy()
        adata_sub = preprocessing_subcluster(
            adata_sub, resolution=resolution, hvg_top_genes=n_top_genes)
        df = _rank_genes_df(adata_sub, "leiden")
        adata_sub.uns["find_markers"] = df
        adata_sub = _run_llm_annotation(
            adata_sub,
            tissue_type=f"{STATE.tissue_type} {cell_type} subtype",
            groupby="leiden"
        )
        key = cell_type.lower().replace(" ", "_")
        STATE.cell_type_adatas[key] = adata_sub
        h5_path = STATE.output_dir / f"subtype_{key}.h5ad"
        adata_sub.write_h5ad(h5_path)
        sc.pl.umap(adata_sub, color=["LLM_annotation"], show=False)
        fig_path = _save_fig(f"umap_subtype_{key}.png")
        vc = adata_sub.obs["LLM_annotation"].value_counts()
        _autosave(f"04_subtype_{cell_type.lower().replace(' ','_')}")
        lines = [f"Subtypes for '{cell_type}' ({mask.sum():,} cells):",
                 f"  h5ad → {h5_path}", f"  UMAP → {fig_path}"]
        for st, n in vc.items():
            lines.append(f"    {st}: {n:,}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_deg_analysis(
    groupby: str = "Disease_type",
    group1: str = "AD",
    group2: str = "PN",
    cell_type: str = "",
) -> str:
    """Differential expression analysis (Wilcoxon) between two groups.

    Args:
        groupby: obs column with group labels (default 'Disease_type').
        group1: Test group (e.g. 'AD').
        group2: Reference group (e.g. 'PN').
        cell_type: Optional - restrict to a specific cell type (partial match).
    """
    if STATE.adata is None:
        return "No data loaded."
    try:
        if cell_type:
            key = cell_type.lower().replace(" ", "_")
            adata = STATE.cell_type_adatas.get(key)
            if adata is None:
                mask = STATE.adata.obs["LLM_annotation"].str.contains(
                    cell_type, case=False, na=False)
                adata = STATE.adata[mask].copy()
        else:
            adata = STATE.adata
        if groupby not in adata.obs.columns:
            return f"Column '{groupby}' not found. Available: {list(adata.obs.columns)}"
        sub = adata[adata.obs[groupby].isin([group1, group2])].copy()
        sc.tl.rank_genes_groups(sub, groupby=groupby, groups=[group1],
                                reference=group2, method="wilcoxon",
                                pts=True, use_raw=False)
        df = sc.get.rank_genes_groups_df(sub, group=group1)
        df = df.rename(columns={"names": "gene", "log2foldchanges": "avg_log2FC"})
        sig = df[df["p_val_adj"] < 0.05].copy()
        up   = sig[sig["avg_log2FC"] >  0.5]
        down = sig[sig["avg_log2FC"] < -0.5]
        g_col = "gene" if "gene" in up.columns else up.columns[0]
        label = f"{'%s_' % cell_type if cell_type else ''}{group1}_vs_{group2}"
        out = STATE.output_dir / f"DEG_{label}.csv"
        sig.to_csv(out, index=False)
        STATE.results["deg"][label] = sig
        return (
            f"DEG: {group1} vs {group2} | context: {cell_type or 'all cells'}\n"
            f"  Sig (padj<0.05)           : {sig.shape[0]}\n"
            f"  Up in {group1} (|FC|>0.5) : {up.shape[0]} | "
            f"top: {up.nlargest(5,'avg_log2FC')[g_col].tolist()}\n"
            f"  Down in {group1}          : {down.shape[0]} | "
            f"top: {down.nsmallest(5,'avg_log2FC')[g_col].tolist()}\n"
            f"  Saved: {out}  (key: '{label}')"
        )
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_deg_all_celltypes(
    groupby: str = "Disease_type",
    group1: str = "AD",
    group2: str = "PN",
    annotation_col: str = "LLM_annotation",
    log2fc_thresh: float = 0.5,
    padj_thresh: float = 0.05,
) -> str:
    """DEG analysis for every cell type separately (group1 vs group2).

    Args:
        groupby: obs column with disease/condition labels.
        group1: Test group.
        group2: Reference group.
        annotation_col: obs column with cell type labels (default 'LLM_annotation').
        log2fc_thresh: |log2FC| threshold (default 0.5).
        padj_thresh: Adjusted p-value threshold (default 0.05).
    """
    if STATE.adata is None:
        return "No data loaded."
    if annotation_col not in STATE.adata.obs.columns:
        return f"'{annotation_col}' not found."
    try:
        adata = STATE.adata
        rows = []
        for ct in adata.obs[annotation_col].unique():
            mask = adata.obs[annotation_col] == ct
            sub = adata[mask & adata.obs[groupby].isin([group1, group2])].copy()
            if sub.obs[groupby].nunique() < 2 or sub.n_obs < 20:
                rows.append({"cell_type": ct, "up": "skipped", "down": ""})
                continue
            sc.tl.rank_genes_groups(sub, groupby=groupby, groups=[group1],
                                    reference=group2, method="wilcoxon",
                                    pts=True, use_raw=False)
            df = sc.get.rank_genes_groups_df(sub, group=group1)
            df = df.rename(columns={"names": "gene", "log2foldchanges": "avg_log2FC"})
            sig = df[(df["p_val_adj"] < padj_thresh) &
                     (df["avg_log2FC"].abs() > log2fc_thresh)]
            label = f"{ct}_{group1}_vs_{group2}"
            STATE.results["deg"][label] = sig
            sig.to_csv(STATE.output_dir / f"DEG_{label}.csv", index=False)
            rows.append({"cell_type": ct,
                         "up":   int((sig["avg_log2FC"] > 0).sum()),
                         "down": int((sig["avg_log2FC"] < 0).sum())})
        summary = pd.DataFrame(rows)
        out = STATE.output_dir / f"DEG_summary_{group1}_vs_{group2}.csv"
        summary.to_csv(out, index=False)
        return (
            f"DEG all cell types ({group1} vs {group2}):\n\n"
            + summary.to_string(index=False)
            + f"\n\nSaved: {out}"
        )
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_pathway_analysis(
    deg_label: str = "",
    gene_list: str = "",
    direction: str = "up",
    top_n: int = 15,
) -> str:
    """GO/KEGG/WikiPathways enrichment analysis.

    Args:
        deg_label: Key from DEG results (use get_session_state() to list keys).
        gene_list: Comma-separated gene list (alternative to deg_label).
        direction: 'up', 'down', or 'both' (default 'up').
        top_n: Top pathways shown in plot (default 15).
    """
    try:
        from pathway_analysis import perform_enrichment_analysis, plot_bubble_enrichment
        if deg_label and deg_label in STATE.results["deg"]:
            df = STATE.results["deg"][deg_label]
            fc_col = "avg_log2FC" if "avg_log2FC" in df.columns else df.columns[1]
            g_col  = "gene"           if "gene"           in df.columns else df.columns[0]
            if direction == "up":
                genes = df[df[fc_col] > 0.25][g_col].dropna().tolist()
            elif direction == "down":
                genes = df[df[fc_col] < -0.25][g_col].dropna().tolist()
            else:
                genes = df[g_col].dropna().tolist()
            label = deg_label
        elif gene_list:
            genes = [g.strip() for g in gene_list.split(",") if g.strip()]
            label = "custom"
        else:
            return f"Provide deg_label or gene_list. Keys: {list(STATE.results['deg'].keys())}"

        genes = [g for g in genes if isinstance(g, str)][:300]
        if not genes:
            return "Gene list is empty."
        enrich = perform_enrichment_analysis(genes, organism="human", adjusted=True)
        out = STATE.output_dir / f"pathway_{label}_{direction}.csv"
        enrich.to_csv(out, index=False)
        STATE.results["enrichment"][f"{label}_{direction}"] = enrich
        fig_path = None
        if not enrich.empty:
            plot_bubble_enrichment(enrich, title=f"Pathway: {label} ({direction})", top_n=top_n)
            fig_path = _save_fig(f"pathway_bubble_{label}_{direction}.png")
        term_col = next((c for c in ["Term","term"] if c in enrich.columns), enrich.columns[0])
        pval_col = next((c for c in ["Adjusted P-value","P-value"] if c in enrich.columns),
                        enrich.columns[-1])
        lines = [f"Pathway enrichment '{label}' ({direction}) — {len(genes)} genes:"]
        for _, row in enrich.head(top_n).iterrows():
            lines.append(f"  {str(row[term_col])[:60]:<60} p={row[pval_col]:.2e}")
        lines.append(f"CSV  : {out}")
        if fig_path:
            lines.append(f"Plot : {fig_path}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_pathway_per_celltype(
    group1: str = "AD",
    group2: str = "PN",
    direction: str = "up",
    top_n: int = 10,
) -> str:
    """Pathway enrichment for each cell type (requires run_deg_all_celltypes first).

    Args:
        group1: Test group used in DEG analysis.
        group2: Reference group used in DEG analysis.
        direction: 'up', 'down', or 'both' (default 'up').
        top_n: Top pathways per cell type (default 10).
    """
    try:
        from pathway_analysis import perform_enrichment_analysis
        pattern = f"_{group1}_vs_{group2}"
        keys = [k for k in STATE.results["deg"] if k.endswith(pattern)]
        if not keys:
            return f"No DEG results for {group1} vs {group2}. Run run_deg_all_celltypes() first."
        rows = []
        for label in keys:
            ct = label.replace(pattern, "")
            df = STATE.results["deg"][label]
            if df.empty:
                continue
            fc_col = "avg_log2FC" if "avg_log2FC" in df.columns else df.columns[1]
            g_col  = "gene"           if "gene"           in df.columns else df.columns[0]
            if direction == "up":
                genes = df[df[fc_col] > 0.25][g_col].dropna().tolist()[:200]
            elif direction == "down":
                genes = df[df[fc_col] < -0.25][g_col].dropna().tolist()[:200]
            else:
                genes = df[g_col].dropna().tolist()[:200]
            if not genes:
                continue
            try:
                enrich = perform_enrichment_analysis(genes, organism="human", adjusted=True)
                key_e = f"{label}_{direction}"
                STATE.results["enrichment"][key_e] = enrich
                enrich.to_csv(STATE.output_dir / f"pathway_{key_e}.csv", index=False)
                term_col = next((c for c in ["Term","term"] if c in enrich.columns), enrich.columns[0])
                pval_col = next((c for c in ["Adjusted P-value","P-value"] if c in enrich.columns),
                                enrich.columns[-1])
                for _, row in enrich.head(top_n).iterrows():
                    rows.append({"cell_type": ct, "term": row[term_col], "pval": row[pval_col]})
            except Exception as ee:
                rows.append({"cell_type": ct, "term": f"ERROR: {ee}", "pval": None})
        if not rows:
            return "No enrichment results."
        out_df = pd.DataFrame(rows)
        out = STATE.output_dir / f"pathway_per_celltype_{group1}_vs_{group2}.csv"
        out_df.to_csv(out, index=False)
        return (
            f"Pathway per cell type ({group1} vs {group2}, {direction}):\n\n"
            + out_df.to_string(index=False)
            + f"\n\nSaved: {out}"
        )
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_composition_analysis(
    groupby: str = "Disease_type",
    annotation_col: str = "LLM_annotation",
    patient_col: str = "patient_final",
) -> str:
    """Analyse cell type composition across patient groups.

    Args:
        groupby: obs column with group labels (default 'Disease_type').
        annotation_col: obs column with cell type labels (default 'LLM_annotation').
        patient_col: obs column with patient IDs for per-patient breakdown.
    """
    if STATE.adata is None:
        return "No data loaded."
    if annotation_col not in STATE.adata.obs.columns:
        return f"'{annotation_col}' not found. Run annotate_cell_types() first."
    try:
        adata = STATE.adata
        prop = (adata.obs.groupby([groupby, annotation_col]).size()
                .reset_index(name="count"))
        total = adata.obs.groupby(groupby).size().reset_index(name="total")
        prop = prop.merge(total, on=groupby)
        prop["proportion"] = prop["count"] / prop["total"]
        pivot = prop.pivot_table(index=annotation_col, columns=groupby,
                                 values="proportion", fill_value=0)
        out_csv = STATE.output_dir / f"composition_{groupby}.csv"
        pivot.to_csv(out_csv)
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.5), 6))
        pivot.T.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_xlabel(groupby); ax.set_ylabel("Proportion")
        ax.set_title(f"Cell Composition by {groupby}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        fig_path = _save_fig(f"composition_{groupby}.png")
        extra = ""
        if patient_col in adata.obs.columns:
            per_pt = (adata.obs.groupby([patient_col, groupby, annotation_col]).size()
                      .reset_index(name="count"))
            pp_tot = adata.obs.groupby([patient_col, groupby]).size().reset_index(name="total")
            per_pt = per_pt.merge(pp_tot, on=[patient_col, groupby])
            per_pt["proportion"] = per_pt["count"] / per_pt["total"]
            out_pt = STATE.output_dir / f"composition_per_patient_{groupby}.csv"
            per_pt.to_csv(out_pt, index=False)
            extra = f"\n  Per-patient CSV: {out_pt}"
        return (
            f"Composition by '{groupby}':\n\n"
            + pivot.round(4).to_string()
            + f"\n\n  CSV  : {out_csv}\n  Plot : {fig_path}{extra}"
        )
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def visualize_markers_dotplot(
    genes: str = "",
    groupby: str = "LLM_annotation",
    top_n: int = 5,
) -> str:
    """Generate a dotplot of marker genes.

    Args:
        genes: Comma-separated gene symbols. If empty, auto-selects top markers.
        groupby: obs column to group by (default 'LLM_annotation').
        top_n: Genes per group when auto-selecting (default 5).
    """
    if STATE.adata is None:
        return "No data loaded."
    try:
        adata = STATE.adata
        if genes:
            gene_list = [g.strip() for g in genes.split(",")]
        else:
            if "find_markers" not in adata.uns:
                return "Run compute_marker_genes() first."
            df = adata.uns["find_markers"]
            g_col   = "gene"           if "gene"           in df.columns else df.columns[1]
            fc_col  = "avg_log2FC" if "avg_log2FC" in df.columns else df.columns[2]
            grp_col = "group"          if "group"          in df.columns else df.columns[0]
            top = df.groupby(grp_col).apply(
                lambda x: x.nlargest(top_n, fc_col)[g_col].tolist())
            gene_list = list(dict.fromkeys(g for gs in top for g in gs))
        gene_list = [g for g in gene_list if g in adata.var_names]
        if not gene_list:
            return "No valid genes to plot."
        sc.pl.dotplot(adata, var_names=gene_list, groupby=groupby, show=False)
        fig_path = _save_fig(f"dotplot_{groupby}.png")
        shown = ", ".join(gene_list[:30]) + ("..." if len(gene_list)>30 else "")
        return f"Dotplot → {fig_path}\nGenes: {shown}"
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def plot_umap(color_by: str = "LLM_annotation") -> str:
    """Plot UMAP colored by a metadata column or gene expression.

    Args:
        color_by: obs column name or gene symbol (default 'LLM_annotation').
    """
    if STATE.adata is None:
        return "No data loaded."
    if "X_umap" not in STATE.adata.obsm:
        return "UMAP not computed. Run preprocess_and_cluster() first."
    try:
        sc.pl.umap(STATE.adata, color=[color_by], show=False)
        fig_path = _save_fig(f"umap_{color_by.replace('/','_')}.png")
        return f"UMAP → {fig_path}"
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def run_trajectory_analysis(
    cell_type: str = "Keratinocyte",
    start_gene: str = "KRT5",
    n_dcs: int = 5,
) -> str:
    """Palantir trajectory inference (diffusion maps -> pseudotime).

    Args:
        cell_type: Cell type to analyse (matched against LLM_annotation).
        start_gene: Gene highly expressed in the progenitor/start state.
        n_dcs: Number of diffusion components (default 5).
    """
    if STATE.adata is None:
        return "No data loaded."
    try:
        import palantir
        import scipy.sparse as sp
        key = cell_type.lower().replace(" ", "_")
        adata = STATE.cell_type_adatas.get(key)
        if adata is None:
            mask = STATE.adata.obs["LLM_annotation"].str.contains(
                cell_type, case=False, na=False)
            if mask.sum() == 0:
                return f"No cells matched '{cell_type}'."
            adata = STATE.adata[mask].copy()
        if adata.X.max() > 50:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=30)
        dm = palantir.utils.run_diffusion_maps(
            pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names),
            n_components=n_dcs)
        ms = palantir.utils.determine_multiscale_space(dm)
        if start_gene in adata.var_names:
            idx = list(adata.var_names).index(start_gene)
            expr = (np.asarray(adata.X[:, idx].todense()).flatten()
                    if sp.issparse(adata.X) else adata.X[:, idx])
            start_cell = adata.obs_names[int(np.argmax(expr))]
        else:
            start_cell = adata.obs_names[0]
        pr = palantir.core.run_palantir(ms, start_cell, num_waypoints=500)
        adata.obs["palantir_pseudotime"] = pr.pseudotime
        adata.obs["palantir_entropy"]    = pr.entropy
        sc.pl.umap(adata, color=["palantir_pseudotime","palantir_entropy"], show=False)
        fig_path = _save_fig(f"trajectory_{key}.png")
        STATE.cell_type_adatas[key] = adata
        h5_path = STATE.output_dir / f"trajectory_{key}.h5ad"
        adata.write_h5ad(h5_path)
        return (
            f"Trajectory '{cell_type}' ({adata.n_obs:,} cells)\n"
            f"  Start cell ({start_gene}): {start_cell}\n"
            f"  Pseudotime: {adata.obs['palantir_pseudotime'].min():.3f} – "
            f"{adata.obs['palantir_pseudotime'].max():.3f}\n"
            f"  Plot → {fig_path}\n  h5ad → {h5_path}"
        )
    except ImportError:
        return "Palantir not installed: pip install palantir"
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@tool
def save_results(output_path: str = "") -> str:
    """Save the current AnnData to disk.

    Args:
        output_path: File path (defaults to <output_dir>/final_results.h5ad).
    """
    if STATE.adata is None:
        return "No data to save."
    path = output_path or str(STATE.output_dir / "final_results.h5ad")
    STATE.adata.write_h5ad(path)
    return f"Saved → {path}"


@tool
def switch_model(
    model_name: str,
    use_openrouter: bool = True,
    openrouter_api_key: str = "",
    openai_api_key: str = "",
) -> str:
    """Switch the LLM used for cell type annotation.

    Args:
        model_name: Short model name. Use list_models() to see all options.
                    Examples: 'gpt-4o', 'claude-3.5-sonnet', 'llama-3.3-70b', 'deepseek-r1'.
        use_openrouter: Route through OpenRouter (default True).
        openrouter_api_key: OpenRouter API key (stored in session).
        openai_api_key: OpenAI API key (stored in session).
    """
    prev = STATE.model_name
    STATE.model_name = model_name
    STATE.use_openrouter = use_openrouter
    if openrouter_api_key:
        STATE.openrouter_api_key = openrouter_api_key
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    if openai_api_key:
        STATE.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    router_id = OPENROUTER_MODELS.get(model_name, model_name)
    via = f"OpenRouter ({router_id})" if use_openrouter else "OpenAI direct"
    return f"Model: {prev} -> {model_name} via {via}"


@tool
def list_models() -> str:
    """List all available models with their OpenRouter IDs."""
    categories = {
        "OpenAI"    : ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-3.5-turbo"],
        "Anthropic" : ["claude-3.5-sonnet","claude-3-opus","claude-3-haiku",
                       "claude-sonnet-4","claude-opus-4"],
        "Meta Llama": ["llama-3.1-70b","llama-3.3-70b","llama-4-maverick","llama-4-scout"],
        "Google"    : ["gemini-pro-1.5","gemini-flash-1.5","gemini-2.0-flash"],
        "Mistral"   : ["mistral-large","mixtral-8x7b"],
        "DeepSeek"  : ["deepseek-v3","deepseek-r1"],
        "Qwen"      : ["qwen-72b"],
    }
    lines = ["Available models:\n"]
    for provider, models in categories.items():
        lines.append(f"  {provider}:")
        for m in models:
            lines.append(f"    {m:<24} -> {OPENROUTER_MODELS[m]}")
    lines.append("\nExample: switch_model('claude-3.5-sonnet', use_openrouter=True)")
    return "\n".join(lines)


@tool
def set_output_dir(path: str) -> str:
    """Set the directory where all output files are saved.

    Args:
        path: Directory path (created if it doesn't exist).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    STATE.output_dir = p
    return f"Output directory -> {p.resolve()}"


# ==============================================================================
# All tools list
# ==============================================================================

ALL_TOOLS = [
    load_h5ad, get_session_state, preprocess_and_cluster,
    compute_marker_genes, annotate_cell_types, annotate_subtypes,
    run_deg_analysis, run_deg_all_celltypes,
    run_pathway_analysis, run_pathway_per_celltype,
    run_composition_analysis, visualize_markers_dotplot, plot_umap,
    run_trajectory_analysis, save_results, switch_model, list_models, set_output_dir,
]

# ==============================================================================
# System prompt
# ==============================================================================

SYSTEM_PROMPT = """\
You are an expert single-cell RNA-seq analysis assistant for skin disease research
(atopic dermatitis / prurigo nodularis / healthy controls).

Tools: load_h5ad, get_session_state, preprocess_and_cluster, compute_marker_genes,
annotate_cell_types, annotate_subtypes, run_deg_analysis, run_deg_all_celltypes,
run_pathway_analysis, run_pathway_per_celltype, run_composition_analysis,
visualize_markers_dotplot, plot_umap, run_trajectory_analysis, save_results,
switch_model (GPT-4o/Claude/Llama/DeepSeek/Gemini via OpenRouter), list_models,
set_output_dir.

Recommended workflow:
load_h5ad -> preprocess_and_cluster -> compute_marker_genes -> annotate_cell_types
-> annotate_subtypes (KC, T cell, Fibroblast...) -> run_deg_all_celltypes
-> run_pathway_per_celltype -> run_composition_analysis -> run_trajectory_analysis -> save_results

Rules:
- Describe what you're about to do BEFORE calling a tool.
- After each result, summarise and ask the user if they want to refine or continue.
- Proactively suggest logical next steps.
- If a prerequisite is missing, explain and offer to fix it.
- Adapt when the user gives feedback (lower resolution, different cell type, etc.).
"""

# ==============================================================================
# Agent factory (LangChain 1.2.x / LangGraph)
# ==============================================================================

# OpenAI-native model shortnames (route directly, not via OpenRouter)
_OPENAI_NATIVE = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}


def _is_openai_native(model_name: str) -> bool:
    return model_name in _OPENAI_NATIVE or model_name.startswith("gpt-")


def _make_llm(model_name, openai_api_key="", openrouter_api_key="",
              use_openrouter=False, temperature=0.0):
    """Build a ChatOpenAI LLM.

    Auto-routing logic:
    - If use_openrouter=True  → always use OpenRouter
    - If model is not OpenAI-native and openrouter_api_key exists → auto OpenRouter
    - Otherwise → OpenAI direct
    """
    # Auto-detect: non-OpenAI model + OpenRouter key available → force OpenRouter
    if not use_openrouter and not _is_openai_native(model_name) and openrouter_api_key:
        use_openrouter = True

    if use_openrouter and openrouter_api_key:
        router_id = OPENROUTER_MODELS.get(model_name, model_name)
        print(f"  [LLM] Model    : {model_name}")
        print(f"  [LLM] Router ID: {router_id}")
        print(f"  [LLM] Endpoint : https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=router_id,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://github.com/scrnaseq-agent",
                "X-Title": "scRNA-seq Analysis Agent",
            },
        )

    key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            f"No API key available for model '{model_name}'. "
            "Pass openai_api_key= or set OPENAI_API_KEY env var."
        )
    print(f"  [LLM] Model    : {model_name}")
    print(f"  [LLM] Endpoint : https://api.openai.com/v1")
    return ChatOpenAI(model=model_name, api_key=key, temperature=temperature)


def create_agent(
    openai_api_key: str = "",
    openrouter_api_key: str = "",
    agent_model: str = "gpt-4o",
    annotation_model: str = "gpt-4o",
    use_openrouter_for_agent: bool = False,
    use_openrouter_for_annotation: bool = False,
):
    """Create the scRNA-seq analysis agent.

    Args:
        openai_api_key: OpenAI API key.
        openrouter_api_key: OpenRouter API key (enables model switching).
        agent_model: Model for agent reasoning/tool-calling
                     (e.g. 'gpt-4o', 'claude-sonnet-4', 'llama-3.3-70b').
        annotation_model: Model for LLM cell type annotation.
        use_openrouter_for_agent: Force OpenRouter for agent LLM.
                                  Auto-enabled when model is non-OpenAI.
        use_openrouter_for_annotation: Force OpenRouter for annotation calls.
                                       Auto-enabled when model is non-OpenAI.

    Returns:
        Compiled LangGraph agent with in-memory conversation history.
    """
    if openai_api_key:
        STATE.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if openrouter_api_key:
        STATE.openrouter_api_key = openrouter_api_key
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    # Auto-enable OpenRouter for annotation when model is non-OpenAI
    if not _is_openai_native(annotation_model) and openrouter_api_key:
        use_openrouter_for_annotation = True

    STATE.model_name     = annotation_model
    STATE.use_openrouter = use_openrouter_for_annotation

    print("\n=== Creating scRNA-seq Agent ===")
    print(f"Agent model      : {agent_model}")
    print(f"Annotation model : {annotation_model}")
    print(f"OpenRouter       : {'yes' if (openrouter_api_key and not _is_openai_native(agent_model)) else 'no (OpenAI direct)'}")
    print()

    try:
        llm = _make_llm(
            model_name=agent_model,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            use_openrouter=use_openrouter_for_agent,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create LLM for agent_model='{agent_model}'.\n"
            f"Tip: non-OpenAI models (Claude, Llama, etc.) require an OpenRouter key.\n"
            f"Available models: {list(OPENROUTER_MODELS.keys())}\n"
            f"Original error: {e}"
        ) from e

    global _AGENT
    _AGENT = _lc_create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    print("\nAgent ready!")
    return _AGENT


# ==============================================================================
# Chat helpers
# ==============================================================================

_CONFIG = {"configurable": {"thread_id": "scrnaseq-session"}}


def chat(agent, message: str, print_output: bool = True) -> str:
    """Send a message to the agent and return the response.

    Args:
        agent: Compiled agent returned by create_agent().
        message: User message string.
        print_output: Print the response (default True).

    Returns:
        Agent response string.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=_CONFIG,
    )
    output = result["messages"][-1].content
    if print_output:
        print(f"\nAgent:\n{output}\n")
    return output


def run_interactive_session(agent) -> None:
    """Terminal-style interactive loop."""
    print("=" * 70)
    print("  scRNA-seq Interactive Analysis Agent  (type 'exit' to quit)")
    print("=" * 70)
    chat(agent, "Hello! Briefly introduce yourself and the analysis you can perform.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended."); break
        if not user_input:
            continue
        if user_input.lower() in {"exit","quit","q"}:
            print("Session ended."); break
        try:
            chat(agent, user_input)
        except Exception as e:
            print(f"\n[Error] {e}\n")


# ==============================================================================
# CLI entry point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="scRNA-seq Analysis Agent")
    p.add_argument("--openai-key",       default=os.environ.get("OPENAI_API_KEY","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"))
    p.add_argument("--openrouter-key",   default=os.environ.get("OPENROUTER_API_KEY","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"))
    p.add_argument("--agent-model",      default="gpt-4o")
    p.add_argument("--annotation-model", default="gpt-4o")
    p.add_argument("--use-openrouter",   action="store_true")
    args = p.parse_args()
    agent = create_agent(
        openai_api_key=args.openai_key,
        openrouter_api_key=args.openrouter_key,
        agent_model=args.agent_model,
        annotation_model=args.annotation_model,
        use_openrouter_for_agent=args.use_openrouter,
        use_openrouter_for_annotation=bool(args.openrouter_key),
    )
    run_interactive_session(agent)
