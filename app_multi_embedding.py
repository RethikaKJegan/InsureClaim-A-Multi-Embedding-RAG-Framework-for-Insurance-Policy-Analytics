# ============================================================
# Multi-Embedding Policy Analysis System with Streamlit (Final Stable Version - No Emojis)
# ============================================================

# --- Prevent Streamlit & torch watcher conflict (set before imports) ---
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("STREAMLIT_WATCHER_TYPE", "poll")

import streamlit as st
from pathlib import Path
import time
from typing import Dict
import plotly.graph_objects as go

# --- Import your project modules ---
from document_processor import DocumentProcessor
from vector_store import SemanticSearchEngine
from answer_generator import GeminiAnswerGenerator, Answer


# ============================================================
# MODEL DEFINITIONS
# ============================================================
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {
        "name": "MPNet Base v2",
        "description": "Best overall quality. 768 dimensions.",
        "dim": 768,
        "speed": "Medium",
        "quality": "Excellent"
    },
    "all-MiniLM-L6-v2": {
        "name": "MiniLM L6 v2",
        "description": "Fast and balanced. 384 dimensions.",
        "dim": 384,
        "speed": "Fast",
        "quality": "Good"
    },
    "multi-qa-mpnet-base-dot-v1": {
        "name": "Multi-QA MPNet",
        "description": "Optimized for Question Answering. 768 dimensions.",
        "dim": 768,
        "speed": "Medium",
        "quality": "Excellent for Q&A"
    }
}


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(page_title="Multi-Embedding Policy Analysis", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0057e7 0%, #00bfa5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .model-card {
        background: linear-gradient(135deg, #0057e7 0%, #00bfa5 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SYSTEM INITIALIZATION
# ============================================================
@st.cache_resource
def initialize_all_systems():
    """Initialize all embedding systems and Gemini models."""
    systems = {}
    progress = st.progress(0)
    status = st.empty()

    total = len(EMBEDDING_MODELS)

    for i, (key, info) in enumerate(EMBEDDING_MODELS.items()):
        status.text(f"Initializing {info['name']} ({i+1}/{total})...")
        try:
            processor = DocumentProcessor(chunk_size=1000, overlap=200)
            search_engine = SemanticSearchEngine(model_name=key)
            generator = GeminiAnswerGenerator(
                api_key="AIzaSyBcWKgR2z_ejabRyNZbEJ236Cs8rm5giMI"
            )
            systems[key] = {
                "doc_processor": processor,
                "search_engine": search_engine,
                "answer_generator": generator,
                "indexed_files": set(),
                "all_chunks": []
            }
        except Exception as e:
            st.error(f"Failed to initialize {info['name']}: {e}")
        progress.progress((i + 1) / total)

    progress.empty()
    status.text("All models initialized successfully.")
    time.sleep(0.5)
    status.empty()
    return systems


# ============================================================
# DOCUMENT PROCESSING
# ============================================================
def process_document_all_models(uploaded_file, systems: Dict):
    """Process once and index across all models."""
    temp_path = Path(f"temp_{uploaded_file.name}")
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        first_sys = next(iter(systems.values()))
        chunks = first_sys["doc_processor"].process_document(str(temp_path))
        if not chunks:
            st.error(f"Failed to extract text from {uploaded_file.name}")
            return False

        st.success(f"Document processed: {len(chunks)} chunks extracted")

        progress = st.progress(0)
        status = st.empty()

        for i, (key, sys_data) in enumerate(systems.items()):
            if uploaded_file.name in sys_data["indexed_files"]:
                continue
            status.text(f"Indexing into {EMBEDDING_MODELS[key]['name']} ({i+1}/{len(systems)})...")
            try:
                sys_data["search_engine"].index_documents(chunks)
                sys_data["all_chunks"].extend(chunks)
                sys_data["indexed_files"].add(uploaded_file.name)
            except Exception as e:
                st.error(f"Indexing error in {key}: {e}")
            progress.progress((i + 1) / len(systems))

        progress.empty()
        status.text("Document indexed across all models.")
        time.sleep(0.5)
        status.empty()
        return True
    except Exception as e:
        st.error(f"Processing error: {e}")
        return False
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ============================================================
# DISPLAY UTILITIES
# ============================================================
def display_answer_compact(answer: Answer, model_key: str):
    """Render answer block neatly."""
    confidence = (answer.confidence or 0.0) * 100
    info = EMBEDDING_MODELS[model_key]

    if confidence >= 70:
        color = "#28a745"
    elif confidence >= 40:
        color = "#ffc107"
    else:
        color = "#dc3545"

    st.markdown(f"### {info['name']}")
    st.info(answer.answer or "No answer generated.")
    st.markdown(f"<p style='text-align:right;color:{color};font-weight:bold;'>Confidence: {confidence:.0f}%</p>", unsafe_allow_html=True)

    if getattr(answer, "evidence", None):
        with st.expander("View Supporting Evidence"):
            for i, ev in enumerate(answer.evidence[:3], 1):
                st.markdown(f"**Source {i}:** {ev.get('source_file','Unknown')} (Page {ev.get('page_number','?')})")
                st.caption(f"Similarity: {ev.get('similarity',0):.3f}")
                snippet = ev.get('text', '')[:250]
                st.text(snippet + "...")
                st.markdown("---")


# ============================================================
# MULTI-MODEL COMPARISON
# ============================================================
def compare_all_models(question: str, systems: Dict):
    st.markdown("## Multi-Model Comparison Results")
    results = []

    progress = st.progress(0)
    for i, (key, sys_data) in enumerate(systems.items()):
        info = EMBEDDING_MODELS[key]
        progress.progress((i + 1) / len(systems))
        with st.spinner(f"Querying {info['name']}..."):
            start = time.time()
            context = sys_data["search_engine"].search(question, k=5)
            ans = sys_data["answer_generator"].generate_answer(question, context, temperature=0.3)
            results.append({
                "model": info["name"],
                "model_key": key,
                "answer": ans,
                "confidence": float(getattr(ans, "confidence", 0.0)),
                "time": time.time() - start
            })
    progress.empty()

    results.sort(key=lambda x: x["confidence"], reverse=True)

    cols = st.columns(len(results))
    for c, r in zip(cols, results):
        with c:
            st.metric(r["model"], f"{r['confidence']*100:.0f}%", f"{r['time']:.2f}s")

    st.markdown("---")
    tabs = st.tabs([r["model"] for r in results])
    for t, r in zip(tabs, results):
        with t:
            display_answer_compact(r["answer"], r["model_key"])

    fig = go.Figure(
        data=[go.Bar(
            x=[r["model"] for r in results],
            y=[r["confidence"]*100 for r in results],
            text=[f"{r['confidence']:.0%}" for r in results],
            textposition="auto",
            marker_color=[
                "#28a745" if r["confidence"] >= 0.7 else "#ffc107" if r["confidence"] >= 0.4 else "#dc3545"
                for r in results
            ]
        )]
    )
    fig.update_layout(title="Confidence Comparison", yaxis_title="Confidence (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)
    best = results[0]
    st.success(f"Best Result: {best['model']} ({best['confidence']:.0%})")


# ============================================================
# MAIN STREAMLIT APP
# ============================================================
def main():
    st.markdown("<div class='main-header'>Multi-Embedding Policy Analysis</div>", unsafe_allow_html=True)
    st.markdown("Upload once and query across all embedding models instantly.")
    st.markdown("---")

    with st.spinner("Initializing models..."):
        systems = initialize_all_systems()
    st.success(f"{len(systems)} embedding models ready.")

    # Sidebar
    with st.sidebar:
        st.header("System Status")
        for key, sys_data in systems.items():
            info = EMBEDDING_MODELS[key]
            st.markdown(f"**{info['name']}**")
            st.write(f"Files: {len(sys_data['indexed_files'])}")
            st.write(f"Chunks: {sys_data['search_engine'].vector_store.index.ntotal}")
            st.markdown("---")

    tab1, tab2 = st.tabs(["Upload and Query", "About"])

    with tab1:
        st.subheader("Upload Policy Documents")
        uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
        if uploaded_files and st.button("Process and Index All", use_container_width=True):
            for f in uploaded_files:
                if f.name not in next(iter(systems.values()))["indexed_files"]:
                    process_document_all_models(f, systems)
                else:
                    st.info(f"{f.name} is already indexed.")

        st.markdown("---")
        st.subheader("Ask a Question")

        first_sys = next(iter(systems.values()))
        if not first_sys["indexed_files"]:
            st.warning("Please upload and index documents first.")
            return

        examples = [
            "What is the insurer's right to cancel the policy?",
            "What are the exclusions under this policy?",
            "Is organ donor treatment covered?",
            "What is the claim settlement process?",
            "What is the waiting period for pre-existing conditions?"
        ]

        selected = st.selectbox("Example Questions:", [""] + examples, index=0)
        question = st.text_area("Your Question:", value=selected, height=100)

        # Persistent model selection fix
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

        if question:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Query Single Model")
                for key, info in EMBEDDING_MODELS.items():
                    if st.button(f"{info['name']}", key=f"btn_{key}", use_container_width=True):
                        st.session_state.selected_model = key
                        st.session_state.query_time = time.time()

                if st.session_state.selected_model:
                    key = st.session_state.selected_model
                    info = EMBEDDING_MODELS[key]
                    sys_data = systems[key]
                    with st.spinner(f"Processing with {info['name']}..."):
                        context = sys_data["search_engine"].search(question, k=5)
                        answer = sys_data["answer_generator"].generate_answer(question, context, temperature=0.3)
                    st.markdown("---")
                    st.success(f"Result from {info['name']}")
                    display_answer_compact(answer, key)
                    if st.button("Clear Selection", use_container_width=True):
                        st.session_state.selected_model = None

            with col2:
                if st.button("Compare All Models", type="primary", use_container_width=True):
                    compare_all_models(question, systems)

    with tab2:
        st.subheader("About This System")
        st.markdown("""
        Key Features
        - Load all embeddings once
        - Upload once, index to all models
        - Instant single or multi-model querying
        - Visual performance comparison
        """)
        for key, info in EMBEDDING_MODELS.items():
            with st.expander(info["name"]):
                st.write(info)


if __name__ == "__main__":
    main()
