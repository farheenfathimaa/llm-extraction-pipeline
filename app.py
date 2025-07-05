import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import custom modules
from pipeline.document_processor import DocumentProcessor
from pipeline.llm_chains import ExtractionPipeline
from pipeline.evaluator import PipelineEvaluator
from config import Config

# Page configuration
st.set_page_config(
    page_title="LLM Information Extraction Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = []
    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False

def main():
    """Main application function"""
    initialize_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔍 LLM Information Extraction Pipeline</h1>
        <p style="color: white; margin: 0; opacity: 0.8;">Chain multiple LLM calls for comprehensive document analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Processing", "🔗 Chain Builder", "📊 Evaluation", "📈 Analytics"])
    
    with tab1:
        document_processing_tab()
    
    with tab2:
        chain_builder_tab()
    
    with tab3:
        evaluation_tab()
    
    with tab4:
        analytics_tab()

def setup_sidebar():
    """Setup sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # API Keys section
    st.sidebar.subheader("API Keys")
    
    # Check for environment variables first
    openai_key = os.getenv('OPENAI_API_KEY', '')
    langchain_key = os.getenv('LANGCHAIN_API_KEY', '')
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", 
        value=openai_key,
        type="password",
        help="Enter your OpenAI API key"
    )
    
    langchain_key = st.sidebar.text_input(
        "LangChain API Key", 
        value=langchain_key,
        type="password",
        help="Enter your LangChain API key for tracing"
    )
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    
    model_provider = st.sidebar.selectbox(
        "Choose LLM Provider",
        ["OpenAI", "Anthropic", "Local Llama"],
        help="Select your preferred LLM provider"
    )
    
    if model_provider == "OpenAI":
        model_name = st.sidebar.selectbox(
            "OpenAI Model",
            ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=1
        )
    elif model_provider == "Anthropic":
        model_name = st.sidebar.selectbox(
            "Anthropic Model",
            ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            index=1
        )
    else:
        model_name = st.sidebar.text_input(
            "Local Model Path",
            value="llama-2-7b-chat.ggmlv3.q4_0.bin"
        )
    
    # Store in session state
    st.session_state.config = {
        'openai_key': openai_key,
        'langchain_key': langchain_key,
        'model_provider': model_provider,
        'model_name': model_name
    }
    
    # Connection status
    if openai_key and langchain_key:
        st.sidebar.success("✅ API Keys Configured")
    else:
        st.sidebar.warning("⚠️ Please configure API keys")

def document_processing_tab():
    """Document processing interface"""
    st.header("📄 Document Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files for processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Process documents
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files)
    
    with col2:
        st.subheader("Processing Options")
        
        extraction_type = st.selectbox(
            "Extraction Type",
            [
                "Policy Conclusions",
                "Research Insights",
                "Key Findings",
                "Recommendations",
                "Custom Extraction"
            ]
        )
        
        if extraction_type == "Custom Extraction":
            custom_prompt = st.text_area(
                "Custom Extraction Prompt",
                placeholder="Enter your custom extraction instructions..."
            )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    # Display results
    if st.session_state.extraction_results:
        st.subheader("📋 Extraction Results")
        display_extraction_results()

def process_documents(uploaded_files):
    """Process uploaded documents through the pipeline"""
    if not validate_configuration():
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pipeline
        pipeline = ExtractionPipeline(st.session_state.config)
        processor = DocumentProcessor()
        
        results = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Save uploaded file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                # Process document
                document_text = processor.process_document(temp_path)
                
                # Run extraction pipeline
                extraction_result = pipeline.extract_insights(
                    document_text,
                    extraction_type="policy_conclusions"
                )
                
                results.append({
                    'filename': file.name,
                    'timestamp': datetime.now().isoformat(),
                    'extraction_result': extraction_result,
                    'document_length': len(document_text),
                    'processing_time': extraction_result.get('processing_time', 0)
                })
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Store results
        st.session_state.extraction_results.extend(results)
        
        status_text.text("✅ Processing complete!")
        st.success(f"Successfully processed {len(results)} documents")
        
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.error("Please check your API keys and model configuration")

def chain_builder_tab():
    """Chain builder interface"""
    st.header("🔗 Chain Builder")
    
    st.markdown("""
    Build custom extraction chains by combining multiple LLM operations.
    Each step in the chain processes the output from the previous step.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Available Operations")
        
        operations = [
            "Document Summarization",
            "Key Insights Extraction",
            "Policy Conclusions",
            "Sentiment Analysis",
            "Entity Recognition",
            "Topic Modeling",
            "Question Answering",
            "Custom Operation"
        ]
        
        selected_ops = st.multiselect(
            "Select Operations",
            operations,
            default=["Document Summarization", "Key Insights Extraction"]
        )
        
        if "Custom Operation" in selected_ops:
            custom_operation = st.text_area(
                "Custom Operation Prompt",
                placeholder="Define your custom operation..."
            )
    
    with col2:
        st.subheader("Chain Configuration")
        
        chain_name = st.text_input(
            "Chain Name",
            value="My Custom Chain"
        )
        
        parallel_processing = st.checkbox(
            "Enable Parallel Processing",
            help="Process operations in parallel where possible"
        )
        
        validation_enabled = st.checkbox(
            "Enable Validation",
            value=True,
            help="Add validation steps between operations"
        )
    
    # Chain visualization
    if selected_ops:
        st.subheader("🔄 Chain Visualization")
        visualize_chain(selected_ops)
    
    # Save chain
    if st.button("💾 Save Chain Configuration"):
        save_chain_config(chain_name, selected_ops, parallel_processing, validation_enabled)

def evaluation_tab():
    """Evaluation interface"""
    st.header("📊 Evaluation Dashboard")
    
    if not st.session_state.extraction_results:
        st.info("No extraction results available. Please process some documents first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Performance Metrics")
        
        evaluator = PipelineEvaluator()
        
        # Calculate metrics
        if st.button("🔍 Evaluate Results"):
            evaluate_pipeline_performance()
    
    with col2:
        st.subheader("Evaluation Settings")
        
        evaluation_criteria = st.multiselect(
            "Select Evaluation Criteria",
            [
                "Accuracy",
                "Completeness",
                "Relevance",
                "Coherence",
                "Processing Speed"
            ],
            default=["Accuracy", "Completeness"]
        )
        
        ground_truth_file = st.file_uploader(
            "Upload Ground Truth (Optional)",
            type=['json', 'csv'],
            help="Upload ground truth data for comparison"
        )
    
    # Display evaluation results
    if st.session_state.evaluation_history:
        st.subheader("📈 Evaluation History")
        display_evaluation_history()

def analytics_tab():
    """Analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.extraction_results:
        st.info("No data available. Please process some documents first.")
        return
    
    # Key metrics
    display_key_metrics()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Time Analysis")
        display_processing_time_chart()
    
    with col2:
        st.subheader("Document Length Distribution")
        display_document_length_chart()
    
    # Detailed analytics
    st.subheader("📊 Detailed Analytics")
    display_detailed_analytics()

def display_extraction_results():
    """Display extraction results in a formatted way"""
    for result in st.session_state.extraction_results[-5:]:  # Show last 5 results
        with st.expander(f"📄 {result['filename']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Extracted Content:**")
                if 'summary' in result['extraction_result']:
                    st.write(result['extraction_result']['summary'])
                if 'insights' in result['extraction_result']:
                    st.write("**Key Insights:**")
                    for insight in result['extraction_result']['insights']:
                        st.write(f"• {insight}")
            
            with col2:
                st.metric("Document Length", f"{result['document_length']} chars")
                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                st.write(f"**Timestamp:** {result['timestamp']}")

def validate_configuration():
    """Validate that necessary configuration is present"""
    config = st.session_state.get('config', {})
    
    if not config.get('openai_key') and not config.get('langchain_key'):
        st.error("Please configure API keys in the sidebar")
        return False
    
    return True

def visualize_chain(operations):
    """Visualize the chain structure"""
    # Create a simple flow diagram
    flow_text = " → ".join(operations)
    st.code(flow_text, language="text")
    
    # Create a more detailed visualization
    fig = go.Figure()
    
    for i, op in enumerate(operations):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            text=op,
            textposition='middle center',
            marker=dict(size=50, color='lightblue'),
            showlegend=False
        ))
        
        if i < len(operations) - 1:
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[0, 0],
                mode='lines',
                line=dict(width=2, color='gray'),
                showlegend=False
            ))
    
    fig.update_layout(
        title="Chain Flow Visualization",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)

def save_chain_config(name, operations, parallel, validation):
    """Save chain configuration"""
    config = {
        'name': name,
        'operations': operations,
        'parallel_processing': parallel,
        'validation_enabled': validation,
        'created_at': datetime.now().isoformat()
    }
    
    # Save to file or session state
    if 'saved_chains' not in st.session_state:
        st.session_state.saved_chains = []
    
    st.session_state.saved_chains.append(config)
    st.success(f"Chain '{name}' saved successfully!")

def evaluate_pipeline_performance():
    """Evaluate pipeline performance"""
    evaluator = PipelineEvaluator()
    
    # Mock evaluation for now
    evaluation_results = {
        'accuracy': 0.85,
        'completeness': 0.78,
        'relevance': 0.92,
        'coherence': 0.88,
        'processing_speed': 2.3,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.evaluation_history.append(evaluation_results)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{evaluation_results['accuracy']:.2%}")
        st.metric("Completeness", f"{evaluation_results['completeness']:.2%}")
    
    with col2:
        st.metric("Relevance", f"{evaluation_results['relevance']:.2%}")
        st.metric("Coherence", f"{evaluation_results['coherence']:.2%}")
    
    with col3:
        st.metric("Processing Speed", f"{evaluation_results['processing_speed']:.1f}s")

def display_evaluation_history():
    """Display evaluation history"""
    df = pd.DataFrame(st.session_state.evaluation_history)
    
    if not df.empty:
        # Create line chart
        fig = px.line(df, 
                     x='timestamp', 
                     y=['accuracy', 'completeness', 'relevance', 'coherence'],
                     title='Evaluation Metrics Over Time')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        st.dataframe(df, use_container_width=True)

def display_key_metrics():
    """Display key performance metrics"""
    if not st.session_state.extraction_results:
        return
    
    results = st.session_state.extraction_results
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Documents", 
            len(results)
        )
    
    with col2:
        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results)
        st.metric(
            "Avg Processing Time", 
            f"{avg_processing_time:.2f}s"
        )
    
    with col3:
        total_chars = sum(r.get('document_length', 0) for r in results)
        st.metric(
            "Total Characters", 
            f"{total_chars:,}"
        )
    
    with col4:
        success_rate = len([r for r in results if r.get('extraction_result')]) / len(results)
        st.metric(
            "Success Rate", 
            f"{success_rate:.1%}"
        )

def display_processing_time_chart():
    """Display processing time analysis chart"""
    if not st.session_state.extraction_results:
        return
    
    results = st.session_state.extraction_results
    
    df = pd.DataFrame([
        {
            'filename': r['filename'],
            'processing_time': r.get('processing_time', 0),
            'document_length': r.get('document_length', 0)
        }
        for r in results
    ])
    
    fig = px.scatter(df, 
                    x='document_length', 
                    y='processing_time',
                    hover_data=['filename'],
                    title='Processing Time vs Document Length')
    
    st.plotly_chart(fig, use_container_width=True)

def display_document_length_chart():
    """Display document length distribution"""
    if not st.session_state.extraction_results:
        return
    
    results = st.session_state.extraction_results
    lengths = [r.get('document_length', 0) for r in results]
    
    fig = px.histogram(x=lengths, 
                      title='Document Length Distribution',
                      labels={'x': 'Document Length (characters)', 'y': 'Count'})
    
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_analytics():
    """Display detailed analytics table"""
    if not st.session_state.extraction_results:
        return
    
    results = st.session_state.extraction_results
    
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Document Length': r.get('document_length', 0),
            'Processing Time (s)': r.get('processing_time', 0),
            'Timestamp': r['timestamp'],
            'Status': 'Success' if r.get('extraction_result') else 'Failed'
        }
        for r in results
    ])
    
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()