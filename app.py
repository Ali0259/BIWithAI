# app.py
import os
import io
import json
import pickle
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# ---------------------------
# Configuration
# ---------------------------
# Keep your Google Drive .xlsx direct-download link (do NOT change)
GDRIVE_FILE_ID = "1_z_nSPqoVZjUXeenOxG9jjjn4Ay6_ZU9"
EXCEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Chunking & embeddings
CHUNK_SIZE = 5                       # number of rows per chunk (optimizeable)
EMBED_MODEL_NAME = "all-mpnet-base-v2"
EMBED_DIM = 768                      # embedding dim for all-mpnet-base-v2
TOP_K = 5                            # retrieval size

# Persistence files (saved in working dir)
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"           # list[str]
CHUNK_META_FILE = "chunk_meta.pkl"   # list[dict]
DF_SNAPSHOT_FILE = "data_snapshot.parquet"

# Groq model
GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------------------
# Initialize clients & models
# ---------------------------
# Groq client (reads key from env)
GROQ_API_KEY = os.environ.get("GROQ_API")
if not GROQ_API_KEY:
    raise RuntimeError("Please set the GROQ_API environment variable before running the app.")
#client = Groq(api_key="GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API"])

def groq_chat(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    # response object uses attribute access
    return resp.choices[0].message.content

# Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# In-memory variables (populated at startup)
faiss_index = None         # faiss index instance
chunks: List[str] = []     # textual chunks
chunk_meta: List[Dict] = []  # metadata per chunk (aggregated numeric fields)
df: pd.DataFrame = None    # loaded dataframe


# ---------------------------
# Utilities: persistence
# ---------------------------
def save_index_and_meta():
    """Save FAISS index + chunks + metadata + df snapshot to disk."""
    global faiss_index, chunks, chunk_meta, df
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    with open(CHUNK_META_FILE, "wb") as f:
        pickle.dump(chunk_meta, f)
    if df is not None:
        df.to_parquet(DF_SNAPSHOT_FILE)


def load_index_and_meta() -> bool:
    """Attempt to load saved index and meta. Returns True if successful."""
    global faiss_index, chunks, chunk_meta, df
    try:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(CHUNK_META_FILE):
            faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            with open(CHUNK_META_FILE, "rb") as f:
                chunk_meta = pickle.load(f)
            if os.path.exists(DF_SNAPSHOT_FILE):
                df = pd.read_parquet(DF_SNAPSHOT_FILE)
            return True
    except Exception as e:
        print("Failed to load persisted index/meta:", e)
    return False


# ---------------------------
# Chunking, embedding & index builder
# ---------------------------
def make_chunk_text(row_slice: pd.DataFrame) -> str:
    """Convert a slice of DataFrame rows to a single chunk text."""
    # join rows with separators and give clear field names for LLM context
    lines = []
    for _, r in row_slice.iterrows():
        parts = []
        for col in row_slice.columns:
            parts.append(f"{col}: {r[col]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def build_chunks_and_index(input_df: pd.DataFrame, chunk_size: int = CHUNK_SIZE, rebuild_index: bool = True):
    """Create chunks, metadata, embeddings and build FAISS index."""
    global chunks, chunk_meta, faiss_index, df
    df = input_df.copy().reset_index(drop=True)

    chunks = []
    chunk_meta = []

    # Create chunks of CHUNK_SIZE rows
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        slice_df = df.iloc[start:end]
        chunk_text = make_chunk_text(slice_df)
        chunks.append(chunk_text)

        # create metadata: aggregated numeric stats and category distribution
        try:
            total_sales = float(slice_df["Sales"].sum())
        except Exception:
            total_sales = 0.0
        try:
            total_profit = float(slice_df["Profit"].sum())
        except Exception:
            total_profit = 0.0
        try:
            total_qty = float(slice_df["Quantity"].sum())
        except Exception:
            total_qty = 0.0
        # category counts
        cat_counts = slice_df["Category"].value_counts().to_dict() if "Category" in slice_df.columns else {}

        chunk_meta.append({
            "start_row": start,
            "end_row": end - 1,
            "total_sales": total_sales,
            "total_profit": total_profit,
            "total_quantity": total_qty,
            "category_counts": cat_counts
        })

    # Encode embeddings for all chunks
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # Normalize vectors for cosine similarity using inner product index
    faiss.normalize_L2(embeddings)

    # Create FAISS index (inner product)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    faiss_index = index

    # persist to disk
    save_index_and_meta()


# ---------------------------
# Data fetcher & refresher
# ---------------------------
def fetch_data_from_gdrive() -> pd.DataFrame:
    """Load Excel from Google Drive direct URL (public view)."""
    # read excel directly from Google Drive link
    df_loaded = pd.read_excel(EXCEL_URL, engine="openpyxl")
    return df_loaded


def refresh_data_and_reindex() -> str:
    """Re-fetch the Excel file from Google Drive and rebuild chunks & index."""
    try:
        new_df = fetch_data_from_gdrive()
        build_chunks_and_index(new_df, chunk_size=CHUNK_SIZE)
        return f"Data refreshed: {len(new_df)} rows loaded, {len(chunks)} chunks created."
    except Exception as e:
        return f"Failed to refresh data: {e}"


# ---------------------------
# RAG pipeline (analytics first)
# ---------------------------
def create_category_chart(retrieved_meta: List[Dict[str, Any]]):
    """Create a matplotlib chart (PIL image) for top categories across retrieved_meta."""
    # aggregate categories
    agg = {}
    for m in retrieved_meta:
        for cat, cnt in m.get("category_counts", {}).items():
            agg[cat] = agg.get(cat, 0) + cnt
    if not agg:
        # fallback: show empty placeholder
        fig, ax = plt.subplots(figsize=(6,3))
        ax.text(0.5, 0.5, "No category data", ha="center", va="center")
        ax.axis("off")
    else:
        categories = list(agg.keys())
        counts = [agg[c] for c in categories]
        # sort top 10
        pairs = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)[:10]
        cats, cnts = zip(*pairs)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(cats, cnts)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Top Categories (retrieved context)")
        plt.tight_layout()

    # convert fig to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def rag_pipeline(query: str, top_k: int = TOP_K):
    """
    RAG pipeline tailored for analytics:
      - Embed query
      - Search FAISS
      - Aggregate retrieved metadata
      - Create an analytics prompt for Groq LLM (strictly based on retrieved context)
      - Produce text answer and a small chart (PIL Image) summarizing category distribution
    """
    global chunks, chunk_meta, faiss_index, df
    if faiss_index is None or len(chunks) == 0:
        return "No data available. Contact admin to refresh data.", None

    # 1. embed query and normalize
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # 2. search
    distances, indices = faiss_index.search(q_emb, top_k)
    idx_list = indices[0].tolist()

    # safety: ensure valid indices
    idx_list = [i for i in idx_list if 0 <= i < len(chunks)]
    if not idx_list:
        return "No relevant context found in dataset.", None

    # 3. prepare context & aggregate meta for chart
    retrieved_texts = [chunks[i] for i in idx_list]
    retrieved_meta = [chunk_meta[i] for i in idx_list]

    # Build a concise context summary for the LLM: include chunk meta (sales/profit/qty) and textual chunk (short)
    context_blocks = []
    for i, t in zip(idx_list, retrieved_texts):
        m = chunk_meta[i]
        # keep chunk text trimmed to first 800 chars to avoid huge prompts
        txt_snip = t if len(t) <= 800 else t[:800] + " ... [truncated]"
        context_blocks.append(f"CHUNK {i} | rows {m['start_row']}-{m['end_row']} | sales:{m['total_sales']:.2f} profit:{m['total_profit']:.2f} qty:{m['total_quantity']:.0f}\n{txt_snip}")

    context_str = "\n\n----\n\n".join(context_blocks)

    # 4. make the LLM prompt (analytics style)
    prompt = f"""
You are an expert retail analyst. Use ONLY the provided context to answer the user's question. Do NOT hallucinate or add facts beyond what the context contains.
Context (chunks):
{context_str}

User question:
{query}

Provide:
- A concise direct answer (2-6 bullet points or short paragraphs).
- Include any calculated totals or simple aggregates present in the context (e.g., total sales, total profit).
- If the user asks for a chartable metric, provide that insight and the chart is attached.
Keep answer short, precise and actionable.
"""
    # 5. call Groq
    try:
        answer = groq_chat(prompt)
    except Exception as e:
        answer = f"LLM error: {e}"

    # 6. create chart image (top categories within retrieved)
    try:
        img = create_category_chart(retrieved_meta)
    except Exception:
        img = None

    return answer.strip(), img


# ---------------------------
# Startup: try to load persisted index, otherwise fetch & build
# ---------------------------
if not load_index_and_meta():
    print("No persisted index found; fetching data and building index...")
    try:
        df_loaded = fetch_data_from_gdrive()
        build_chunks_and_index(df_loaded, chunk_size=CHUNK_SIZE)
        print(f"Built index: {len(chunks)} chunks.")
    except Exception as e:
        print("Failed initial build:", e)


# ---------------------------
# Gradio UI (minimal analytics)
# ---------------------------
with gr.Blocks() as ui:
    gr.Markdown("## Inventory & Sales Analytics — Groq + RAG (Analytics-only)")
    gr.Markdown("Ask questions about sales, inventory, restocking suggestions. App will only use dataset from the linked Excel sheet.")

    text_query = gr.Textbox(label="Ask anything about sales or inventory (analytics-only)", placeholder="e.g. 'Which categories need restock in Model Town this month?'")
    output_text = gr.Textbox(label="AI Response", lines=16)
    output_image = gr.Image(label="Contextual Chart (top categories from retrieved context)", type="pil")

    text_query.submit(rag_pipeline, inputs=text_query, outputs=[output_text, output_image])
# ---------------------------
# Add Refresh Button to Gradio UI
# ---------------------------
#with gr.Row():
#    refresh_btn = gr.Button("🔄 Refresh Dataset & Rebuild Index", variant="secondary")
#    refresh_status = gr.Textbox(label="Refresh Status", lines=3)
#
# When clicked, run the refresh and show output
#refresh_btn.click(refresh_data_and_reindex, inputs=None, outputs=refresh_status)

# Launch (no share=True for spaces; keep share for local/colab testing if desired)
if __name__ == "__main__":
    ui.launch()
