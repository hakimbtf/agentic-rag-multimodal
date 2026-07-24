import streamlit as st
from agent_graph import AgenticRAGGraph

st.set_page_config(
    page_title="Agentic RAG & Multimodal AI Pipeline - Abdelhakim Boutafi",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .node-box {
        background: #151C2C;
        border-left: 4px solid #00F2FE;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 Agentic RAG & Multimodal Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Multi-Agent Orchestration • Hybrid Vector Index (Dense + BM25) • Self-Correction RAG Evaluation</div>', unsafe_allow_html=True)

# Initialize Agentic RAG Engine
if "graph" not in st.session_state:
    st.session_state.graph = AgenticRAGGraph()

# Sidebar: Document Ingest & Settings
with st.sidebar:
    st.header("📄 Knowledge Base & Upload")
    uploaded_file = st.file_uploader("Upload Document (PDF / TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.graph.vector_store.add_document(uploaded_file.name, content)
        st.success(f"Indexed {uploaded_file.name} into Hybrid Vector Index!")

    st.subheader("📚 Active Document Corpus")
    chunks = st.session_state.graph.vector_store.chunks
    docs_set = set(c.doc_name for c in chunks)
    for d in docs_set:
        st.markdown(f"- 📄 `{d}`")

# Main Interface: Query Selection & Input
st.subheader("🔍 Ask a Multimodal / Industrial Query")

preset_queries = [
    "How does the GSK Continuous Improvement Agent (CIA) handle model retraining drift?",
    "What are the primary GHG emissions and energy consumption figures for Aluminium production?",
    "How does HPS Switch score 1,000,000 daily financial fraud transactions on Kubernetes?"
]

selected_preset = st.selectbox("Select Sample Enterprise Query or Type Custom Below:", ["-- Select Sample Query --"] + preset_queries)
user_input = st.text_input("Enter Query:", value="" if selected_preset.startswith("--") else selected_preset)

if st.button("🚀 Execute Multi-Agent Graph Query", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Orchestrating LangGraph State Machine..."):
            result = st.session_state.graph.execute_query(user_input)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Execution Latency", f"{result['latency_ms']} ms")
        with col2:
            st.metric("RAG Faithfulness Score", f"{result['ragas_scores']['faithfulness']*100:.1f}%")
        with col3:
            st.metric("Context Relevance", f"{result['ragas_scores']['relevance']*100:.1f}%")

        st.markdown("### 💡 Synthesized Answer")
        st.info(result['final_answer'])

        if result['citations']:
            st.markdown("**📌 Verified Document Citations:**")
            for cite in result['citations']:
                st.markdown(f"- 📄 `{cite}`")

        st.markdown("### 🧬 LangGraph Agent Trajectory Trace")
        for step in result['trajectory']:
            with st.expander(f"Step {step['step']}: {step['node']} ({step['status']})", expanded=True):
                st.write(step['output'])
                if "data" in step:
                    st.json(step['data'])
                if "metrics" in step:
                    st.json(step['metrics'])
