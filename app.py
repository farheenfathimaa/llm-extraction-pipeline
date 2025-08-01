import streamlit as st
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from pipeline.document_processor import DocumentProcessor
from pipeline.llm_chains import ExtractionPipeline
from pipeline.evaluation import EvaluationMetrics as PipelineEvaluator

###############################################################################
# 🔧 Page‑wide settings & Styles
###############################################################################

st.set_page_config(
    page_title="LLM Information Extraction Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Simple CSS tweaks -------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg,#667eea 0%,#764ba2 100%);
            padding:1rem;border-radius:10px;margin-bottom:2rem;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg,#667eea 0%,#764ba2 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# 🧰 Helper functions
###############################################################################


def _init_session():
    """Ensure all the keys we use later exist in session_state."""
    defaults = {
        "extraction_results": [],
        "evaluation_history": [],
        "config": {},
        "saved_chains": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _config_complete() -> bool:
    # Check if the API key is set for the selected provider
    if st.session_state.get("config", {}).get("provider") == "OpenAI" and not st.session_state.get("config", {}).get("openai_key"):
        st.error("🚨 Please provide your OpenAI key in the sidebar.")
        return False
    if st.session_state.get("config", {}).get("provider") == "Anthropic" and not st.session_state.get("config", {}).get("anthropic_key"):
        st.error("🚨 Please provide your Anthropic key in the sidebar.")
        return False
    return True


###############################################################################
# 🖼  Sidebar – API keys & Model selection
###############################################################################

def sidebar():
    st.sidebar.header("⚙️  Configuration")

    # -- Model provider selection --------------------------------------------
    provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Anthropic"],
        index=0,
    )

    st.session_state["config"]["provider"] = provider

    if provider == "OpenAI":
        # -- OpenAI API keys & Model selection --------------------------------
        openai_key_from_env = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=st.session_state.get("config", {}).get("openai_key", openai_key_from_env),
            type="password",
        )
        model_name = st.sidebar.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
        st.session_state["config"]["openai_key"] = openai_key
        st.session_state["config"]["model_name"] = model_name
        if openai_key:
            st.sidebar.success("✅ OpenAI key stored in memory.")

    elif provider == "Anthropic":
        # -- Anthropic API keys & Model selection ------------------------------
        anthropic_key_from_env = os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_key = st.sidebar.text_input(
            "Anthropic API Key",
            value=st.session_state.get("config", {}).get("anthropic_key", anthropic_key_from_env),
            type="password",
        )
        model_name = st.sidebar.selectbox(
            "Anthropic Model",
            ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            index=1,
        )
        st.session_state["config"]["anthropic_key"] = anthropic_key
        st.session_state["config"]["model_name"] = model_name
        if anthropic_key:
            st.sidebar.success("✅ Anthropic key stored in memory.")


###############################################################################
# 📄 Tab 1 – Document processing
###############################################################################


def tab_documents():
    st.header("📄 Document Processing")

    col_left, col_right = st.columns([2, 1])

    # ------------------------------------------------------------------- left
    with col_left:
        uploads = st.file_uploader(
            "Choose files (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploads:
            st.success(f"Uploaded {len(uploads)} file(s)")

    # ------------------------------------------------------------------ right
    with col_right:
        extraction_type = st.selectbox(
            "Extraction type",
            [
                "Policy Conclusions",
                "Research Insights",
                "Key Findings",
                "Recommendations",
                "Custom Extraction",
            ],
        )
        custom_prompt = ""
        if extraction_type == "Custom Extraction":
            custom_prompt = st.text_area("Custom prompt")
        threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)

    # ----------------------------------------------------------------- action
    if uploads and st.button("🚀 Process Documents", use_container_width=True):
        if not _config_complete():
            return
        _run_pipeline(uploads, extraction_type, custom_prompt, threshold)

    # ---------------------------------------------------------------- results
    if st.session_state["extraction_results"]:
        st.subheader("Latest results")
        for res in st.session_state["extraction_results"][-5:]:
            with st.expander(res["filename"], expanded=False):
                st.write(res["extraction_result"].get("summary", "― no summary ―"))
                if insights := res["extraction_result"].get("insights"):
                    st.markdown("**Key insights**")
                    for i in insights:
                        st.write("•", i)
                st.caption(f"Processed {res['document_length']} chars in {res['processing_time']:.2f}s – {res['timestamp']}")

###############################################################################
# 🏃‍♀️  Actual processing work
###############################################################################


def _run_pipeline(files, extraction_type, custom_prompt, threshold):
    doc_processor = DocumentProcessor()
    pipeline = ExtractionPipeline(st.session_state["config"])

    results = []
    prog = st.progress(0)
    stat = st.empty()

    for idx, f in enumerate(files, start=1):
        stat.text(f"Processing {f.name} …")
        prog.progress(idx / len(files))

        tmp = Path(f"tmp__{f.name}")
        tmp.write_bytes(f.read())
        try:
            text = doc_processor.process_document(str(tmp))

            # ------------------------------------------------------------------
            # New call style: ExtractionPipeline.extract_insights(text, task, ...)
            # ------------------------------------------------------------------
            out = pipeline.extract_insights(
                text,
                extraction_type=extraction_type.lower().replace(" ", "_"),
                custom_prompt=custom_prompt or None,
                confidence_threshold=threshold,
            )
            
            results.append(
                {
                    "filename": f.name,
                    "timestamp": datetime.now().isoformat(),
                    "extraction_result": out,
                    "document_length": len(text),
                    "processing_time": out.get("processing_time", 0),
                }
            )
        except Exception as e:
            st.error(f"❌ {f.name}: {e}")
            results.append(
                {
                    "filename": f.name,
                    "timestamp": datetime.now().isoformat(),
                    "extraction_result": {"summary": f"Error: {e}"},
                    "document_length": 0,
                    "processing_time": 0,
                }
            )
        finally:
            tmp.unlink(missing_ok=True)

    st.session_state["extraction_results"].extend(results)
    stat.text("✅ Done!")
    st.success(f"Processed {len(results)} document(s)")

###############################################################################
# 🔗 Tab 2 – Chain builder
###############################################################################

def tab_chain_builder():
    st.header("🔗 Chain Builder")
    st.info("Create and save custom extraction prompts (Chains).")

    chain_name = st.text_input("Chain Name")
    chain_prompt = st.text_area(
        "Chain Prompt (Use {document} as a placeholder for the document text)"
    )

    if st.button("Save Chain"):
        if chain_name and chain_prompt:
            new_chain = {"name": chain_name, "prompt": chain_prompt}
            st.session_state["saved_chains"].append(new_chain)
            st.success(f"Chain '{chain_name}' saved!")
        else:
            st.error("Please provide both a chain name and a prompt.")

    st.subheader("Saved Chains")
    if st.session_state["saved_chains"]:
        for i, chain in enumerate(st.session_state["saved_chains"]):
            with st.expander(chain["name"]):
                st.code(chain["prompt"])

                if st.button(f"Delete '{chain['name']}'", key=f"delete_chain_{i}"):
                    st.session_state["saved_chains"].pop(i)
                    st.experimental_rerun()
    else:
        st.info("No chains saved yet.")

###############################################################################
# 📊 Tab 3 – Evaluation (unchanged except for safety checks)
###############################################################################

def tab_evaluation():
    st.header("📊 Evaluation Dashboard")
    if not st.session_state["extraction_results"]:
        st.info("Run some documents first!")
        return
    if st.button("Run mock evaluation"):
        st.session_state["evaluation_history"].append(
            {
                "accuracy": 0.88,
                "completeness": 0.82,
                "timestamp": datetime.now().isoformat(),
            }
        )
    if st.session_state["evaluation_history"]:
        df = pd.DataFrame(st.session_state["evaluation_history"])
        st.dataframe(df)

###############################################################################
# 📈 Tab 4 – Analytics (trimmed)
###############################################################################

def tab_analytics():
    st.header("📈 Analytics")
    if not st.session_state["extraction_results"]:
        st.info("No data yet")
        return
    res = st.session_state["extraction_results"]
    avg_time = sum(r["processing_time"] for r in res) / len(res)
    st.metric("Average processing time", f"{avg_time:.2f}s")

###############################################################################
# 🚀 Main
###############################################################################

def main():
    _init_session()
    sidebar()
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Documents",
        "🔗 Chains",
        "📊 Evaluation",
        "📈 Analytics",
    ])
    with tab1:
        tab_documents()
    with tab2:
        tab_chain_builder()
    with tab3:
        tab_evaluation()
    with tab4:
        tab_analytics()


if __name__ == "__main__":
    main()
