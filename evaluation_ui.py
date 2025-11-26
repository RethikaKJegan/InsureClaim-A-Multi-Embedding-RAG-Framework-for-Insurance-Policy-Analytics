"""
Interactive Streamlit UI for RAG Evaluation
Beautiful interface to test and evaluate your RAG system
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time

# Import evaluation components
from production_rag_system import ProductionRAGSystem
from generate_gold_dataset import GoldDatasetGenerator
from rag_evaluator import NLPMetricsCalculator
from hf_llm_judge import HuggingFaceLLMJudge

# Page config
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #ffc107;
        font-weight: bold;
    }
    .score-poor {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'gold_dataset' not in st.session_state:
    st.session_state.gold_dataset = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'nlp_calculator' not in st.session_state:
    st.session_state.nlp_calculator = NLPMetricsCalculator()
if 'llm_judge' not in st.session_state:
    st.session_state.llm_judge = None


def initialize_system():
    """Initialize RAG system"""
    with st.spinner("Initializing RAG System..."):
        try:
            system = ProductionRAGSystem(   
                gemini_api_key="AIzaSyBcWKgR2z_ejabRyNZbEJ236Cs8rm5giMI",  # Not needed for HF
                chunk_size=1000,
                overlap=200
            )
            
            # Index documents if available
            if Path("policy1.pdf").exists():
                system.process_and_index("policy1.pdf")
            if Path("policy2.pdf").exists():
                system.process_and_index("policy2.pdf")
            
            st.session_state.rag_system = system
            
            return True
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return False


def load_gold_dataset():
    """Load or generate gold dataset"""
    if Path("gold_test_dataset.json").exists():
        with open("gold_test_dataset.json", 'r') as f:
            dataset = json.load(f)
        st.session_state.gold_dataset = dataset
        return True
    else:
        with st.spinner("Generating gold test dataset..."):
            generator = GoldDatasetGenerator()
            dataset = generator.generate_complete_dataset()
            st.session_state.gold_dataset = dataset
            return True


def evaluate_single_question(question: str, ground_truth: str):
    """Evaluate a single question"""
    
    if not st.session_state.rag_system:
        st.error("Please initialize the system first!")
        return None
    
    # Get RAG answer
    with st.spinner("Thinking..."):
        try:
            rag_answer = st.session_state.rag_system.ask(question)
            predicted = rag_answer.answer
            evidence = rag_answer.evidence
        except Exception as e:
            st.error(f"RAG system error: {e}")
            predicted = ""
            evidence = []
    
    # Calculate NLP metrics
    with st.spinner("Calculating metrics..."):
        nlp_metrics = st.session_state.nlp_calculator.calculate_all_metrics(
            ground_truth,
            predicted
        )
    
    return {
        'question': question,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'evidence': evidence,
        'nlp_metrics': nlp_metrics,
        'llm_scores': {}
    }


def display_single_evaluation(result):
    """Display results for single evaluation"""
    
    st.markdown("---")
    
    # Question and Answers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Question")
        st.info(result['question'])
        
        st.markdown("### Ground Truth")
        st.success(result['ground_truth'])
    
    with col2:
        st.markdown("### RAG Answer")
        st.warning(result['predicted'])
    
    # Metrics Dashboard
    st.markdown("---")
    st.markdown("### Evaluation Metrics")
    
    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)
    
    nlp = result['nlp_metrics']
    
    with col1:
        bleu_4 = nlp['bleu_4']
        color = 'excellent' if bleu_4 > 0.3 else 'good' if bleu_4 > 0.15 else 'poor'
        st.markdown(f"""
        <div class='metric-box'>
            <h3>BLEU-4</h3>
            <h1 class='score-{color}'>{bleu_4:.3f}</h1>
            <p>Word Overlap</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rouge_l = nlp['rouge_l_f']
        color = 'excellent' if rouge_l > 0.5 else 'good' if rouge_l > 0.3 else 'poor'
        st.markdown(f"""
        <div class='metric-box'>
            <h3>ROUGE-L</h3>
            <h1 class='score-{color}'>{rouge_l:.3f}</h1>
            <p>Recall Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        meteor = nlp['meteor']
        color = 'excellent' if meteor > 0.4 else 'good' if meteor > 0.25 else 'poor'
        st.markdown(f"""
        <div class='metric-box'>
            <h3>METEOR</h3>
            <h1 class='score-{color}'>{meteor:.3f}</h1>
            <p>Semantic Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Metrics
    st.markdown("---")
    st.markdown("#### NLP Metrics Breakdown")
    metrics_df = pd.DataFrame({
        'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
                  'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR'],
        'Score': [
            nlp['bleu_1'], nlp['bleu_2'], nlp['bleu_3'], nlp['bleu_4'],
            nlp['rouge_1_f'], nlp['rouge_2_f'], nlp['rouge_l_f'], nlp['meteor']
        ]
    })
    
    fig = px.bar(metrics_df, x='Metric', y='Score', 
                 color='Score', color_continuous_scale='RdYlGn',
                 range_color=[0, 1])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Evidence/Sources
    if result['evidence']:
        with st.expander("Retrieved Evidence", expanded=False):
            for i, ev in enumerate(result['evidence'][:3], 1):
                st.markdown(f"**Source {i}**: {ev.get('source_file', 'Unknown')}")
                if ev.get('similarity'):
                    st.caption(f"Similarity: {ev['similarity']:.3f}")
                st.text_area(f"Chunk {i}", ev.get('text', ''), height=100, key=f"ev_{i}")


def main():
    """Main app"""
    
    # Header
    st.markdown("<div class='main-header'>RAG Evaluation Dashboard</div>", 
                unsafe_allow_html=True)
    st.markdown("**Interactive evaluation of your Production RAG System**")
    
    # Sidebar
    with st.sidebar:
        st.header("System Control")
        
        if st.session_state.rag_system is None:
            if st.button("Initialize System", use_container_width=True):
                if initialize_system():
                    st.rerun()
        else:
            st.success("System Ready")
            stats = st.session_state.rag_system.get_stats()
            st.metric("Indexed Chunks", stats['total_chunks'])
            st.metric("Files", stats['indexed_files'])
        
        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)")
        st.markdown("- [ROUGE Score](https://en.wikipedia.org/wiki/ROUGE_(metric))")
        st.markdown("- [METEOR](https://www.cs.cmu.edu/~alavie/METEOR/)")
    
    # Main content
    st.markdown("## Test Single Question")
    
    # Custom question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="E.g., What is the waiting period for pre-existing diseases?"
    )
    
    ground_truth = st.text_area(
        "Ground truth answer (for comparison):",
        height=100,
        placeholder="Enter the correct answer to compare against..."
    )
    
    if st.button("Evaluate", type="primary"):
        if not question or not ground_truth:
            st.warning("Please provide both question and ground truth!")
        elif not st.session_state.rag_system:
            st.error("Please initialize the system first!")
        else:
            result = evaluate_single_question(question, ground_truth)
            if result:
                display_single_evaluation(result)


if __name__ == "__main__":
    main()