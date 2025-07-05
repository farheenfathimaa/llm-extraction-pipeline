# LLM-Powered Information Extraction Pipeline

A comprehensive pipeline that chains multiple LLM calls to extract policy conclusions and insights from long research documents, demonstrating process supervision rather than single-output reliance.

## Features

- **Multi-LLM Support**: Works with OpenAI GPT, Anthropic Claude, and Hugging Face models
- **Document Processing**: Supports PDF, DOCX, TXT, and Markdown files
- **Chained Analysis**: Implements summarization → key insights → policy conclusions → validation workflow
- **Interactive UI**: Streamlit-based web interface for easy use
- **Evaluation Metrics**: Built-in evaluation system with coherence, relevance, and completeness scoring
- **Export Options**: Results can be exported to JSON, CSV, and Markdown formats
- **Caching System**: Intelligent caching to avoid redundant API calls
- **Real-time Monitoring**: Progress tracking and cost estimation

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (required)
- Anthropic API key (optional)
- VS Code (recommended) or Google Colab

### Installation

1. **Clone or download the project files**
2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Configure API keys**:
   - Edit the `.env` file
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Run the application**:
   ```bash
   python run.py
   ```
   Or alternatively:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** at `http://localhost:8501`

## Project Structure

```
llm-extraction-pipeline/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── setup.py                   # Setup script
├── run.py                     # Application runner
├── .env                       # Environment variables (create this)
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── pipeline/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing logic
│   ├── llm_chains.py         # LLM chain management
│   ├── evaluation.py         # Evaluation metrics
│   └── utils.py              # Utility functions
└── data/
    ├── uploads/              # Uploaded documents
    ├── outputs/              # Generated results
    ├── cache/                # Cached responses
    ├── logs/                 # Application logs
    └── sample_documents/     # Sample test documents
```

## Usage Guide

### 1. Document Upload
- Upload PDF, DOCX, TXT, or Markdown files
- Maximum file size: 10MB
- Multiple documents can be processed in batch

### 2. Configure Processing
- **LLM Model**: Choose from available models (GPT-3.5, GPT-4, Claude, etc.)
- **Chain Type**: Select analysis type (comprehensive, research, policy, etc.)
- **Processing Options**: 
  - Chunk size for large documents
  - Temperature for creativity control
  - Maximum tokens per request

### 3. Run Analysis
- Click "Start Processing" to begin the pipeline
- Monitor progress in real-time
- View intermediate results for each step

### 4. Review Results
- **Summary**: Condensed version of the document
- **Key Insights**: Important findings and patterns
- **Policy Conclusions**: Actionable recommendations
- **Validation**: Quality assessment of the analysis

### 5. Export Results
- Download results in JSON, CSV, or Markdown format
- Save processed documents for future reference
- Export evaluation metrics for quality assessment

## Chain Types

### Comprehensive Analysis
- Full document summarization
- Key insight extraction
- Policy conclusion generation
- Validation and quality checking

### Research Pipeline
- Literature review summary
- Methodology analysis
- Findings extraction
- Research gap identification

### Policy Analysis
- Policy option identification
- Impact assessment
- Implementation feasibility
- Stakeholder analysis

### Quick Insights
- Rapid key point extraction
- Executive summary generation
- Action item identification

## API Configuration

### OpenAI Models
```python
# Supported models
- gpt-3.5-turbo (recommended for cost-effectiveness)
- gpt-4 (higher quality, more expensive)
- gpt-4-turbo (balanced performance)
```

### Anthropic Models
```python
# Supported models
- claude-3-haiku (fast and cost-effective)
- claude-3-sonnet (balanced performance)
- claude-3-opus (highest quality)
```

### Cost Management
- Real-time cost estimation
- Token usage tracking
- Budget alerts and limits
- Caching to reduce API calls

## Evaluation Metrics

The pipeline includes comprehensive evaluation:

- **Coherence**: Logical flow and consistency
- **Relevance**: Alignment with document content
- **Completeness**: Coverage of important topics
- **Accuracy**: Factual correctness
- **Usefulness**: Practical value of insights
- **Cost Efficiency**: Results quality per dollar spent

## Advanced Features

### Custom Chain Development
Create your own processing chains:

```python
from pipeline.llm_chains import LLMChainManager

# Define custom chain
custom_chain = {
    "name": "Custom Analysis",
    "steps": [
        {"type": "summarization", "prompt": "custom_prompt"},
        {"type": "analysis", "prompt": "analysis_prompt"},
        {"type": "validation", "prompt": "validation_prompt"}
    ]
}

# Register and use
chain_manager = LLMChainManager()
chain_manager.register_chain(custom_chain)
```

### Batch Processing
Process multiple documents efficiently:

```python
from pipeline.document_processor import DocumentProcessor

processor = DocumentProcessor()
results = processor.process_batch(document_list, chain_type="comprehensive")
```

### Custom Evaluation
Implement custom evaluation metrics:

```python
from pipeline.evaluation import EvaluationSystem

evaluator = EvaluationSystem()
evaluator.add_custom_metric("domain_specific", custom_metric_function)
```

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py --logger.level=debug
```

### Production Deployment
```bash
# Set production environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501

# Run with production settings
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Check `.env` file exists and contains your API key
   - Verify the key is correctly formatted

2. **Document Processing Errors**
   - Ensure document is not corrupted
   - Check file size limits
   - Verify file format is supported

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Enable document caching
   - Process documents individually

4. **Slow Processing**
   - Use faster models (GPT-3.5 vs GPT-4)
   - Reduce chunk overlap
   - Enable parallel processing

### Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug

# Check logs
tail -f data/logs/pipeline_*.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the application logs
- Create an issue in the repository

## Changelog

### Version 1.0.0
- Initial release
- Multi-LLM support
- Comprehensive pipeline implementation
- Streamlit UI
- Evaluation system
- Export functionality

## Future Enhancements

- [ ] Integration with more LLM providers
- [ ] Advanced document parsing (tables, images)
- [ ] Multi-language support
- [ ] API endpoint for programmatic access
- [ ] Advanced visualization features
- [ ] Collaborative workspace features
- [ ] Integration with document management systems