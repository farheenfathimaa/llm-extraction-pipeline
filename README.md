# LLM-Powered Information Extraction Pipeline

A comprehensive pipeline that chains multiple LLM calls to extract policy conclusions and insights from long research documents, demonstrating process supervision rather than single-output reliance.

## Features

-   **Multi-LLM Support**: Works with OpenAI GPT and Anthropic Claude models.
-   **Document Processing**: Supports PDF, DOCX, TXT, and Markdown files.
-   **Configurable Extraction Tasks**: Implements specific tasks like summarization, key insights, policy conclusions, and validation.
-   **Custom Prompt Builder**: Define and save your own extraction prompts directly within the UI.
-   **Interactive UI**: Streamlit-based web interface for easy use.
-   **Evaluation Metrics**: Built-in system for evaluating pipeline performance and output quality.
-   **Export Options**: Results can be exported to JSON, CSV, and Markdown formats.
-   **Caching System**: Intelligent file-based caching to avoid redundant API calls and save costs.
-   **Real-time Monitoring**: Progress tracking and estimated processing time per document.
-   **Robust Error Handling**: Gracefully handles API rate limits and quota exhaustion, providing clear user feedback.

## Quick Start

### Prerequisites

-   Python 3.8 or higher
-   An OpenAI API key or an Anthropic API key (at least one is required for LLM functionality)
-   VS Code (recommended) or Google Colab

### Installation

1.  **Clone or download the project files**
2.  **Run the setup script**:
    ```bash
    python setup.py
    ```

3.  **Configure API keys**:
    -   Edit the `.env` file
    -   Add your API keys. Example:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ANTHROPIC_API_KEY=your_anthropic_api_key_here
        ```

4.  **Run the application**:
    ```bash
    python run.py
    ```
    Or alternatively:
    ```bash
    streamlit run app.py
    ```

5.  **Open your browser** at `http://localhost:8501`

## Project Structure

````

llm-extraction-pipeline/
├── app.py                      \# Main Streamlit application
├── config.py                   \# Configuration settings
├── requirements.txt            \# Python dependencies
├── setup.py                    \# Setup script
├── run.py                      \# Application runner
├── .env                        \# Environment variables (create this)
├── .gitignore                  \# Git ignore file
├── README.md                   \# This file
├── pipeline/
│   ├── **init**.py
│   ├── document\_processor.py   \# Document processing logic
│   ├── llm\_chains.py           \# LLM chain management and task definitions
│   ├── evaluation.py           \# Evaluation metrics logic
│   ├── cache\_manager.py        \# Specific cache management for LLM calls
│   └── utils.py                \# General utility functions (file, text, export)
└── data/
├── uploads/                \# Uploaded documents
├── outputs/                \# Generated results
├── cache/                  \# Cached LLM responses
├── logs/                   \# Application logs
└── sample\_documents/       \# Sample test documents

````

## Usage Guide

### 1. Document Upload
-   Upload PDF, DOCX, TXT, or Markdown files.
-   Maximum file size: 10MB (configurable).
-   Multiple documents can be processed in a batch.

### 2. Configure Processing
-   **LLM Provider & Model**: Select your desired LLM provider (OpenAI or Anthropic) and model in the sidebar.
-   **Extraction Type**: Choose a predefined task (e.g., Policy Conclusions, Research Insights, Key Findings, Recommendations) or select "Custom Extraction" to use your own prompt.
-   **Processing Options**: Adjust parameters like confidence threshold for certain tasks.

### 3. Run Analysis
-   Click "🚀 Process Documents" to begin the pipeline.
-   Monitor progress in real-time with status updates.
-   View results for each document as they complete, including any errors.

### 4. Review Results
-   **Summary**: A concise overview of the document.
-   **Key Insights**: Important findings and patterns extracted.
-   **Policy Conclusions**: Actionable recommendations derived (for relevant extraction types).
-   **Validation**: Quality assessment (if a validation chain is configured/run).

### 5. Export Results
-   Download results in JSON, CSV, or Markdown format.
-   Output includes extracted text, metadata, and processing details.

## API Configuration

### OpenAI Models
-   `gpt-3.5-turbo` (cost-effective)
-   `gpt-4o` (high-quality, faster than GPT-4, balanced cost)
-   `gpt-4` (higher quality, more expensive)

### Anthropic Models
-   `claude-3-haiku-20240307` (fast and cost-effective)
-   `claude-3-sonnet-20240229` (balanced performance)
-   `claude-3-opus-20240229` (highest quality)

### Cost Management
-   **Intelligent Caching:** Reduces repeated LLM API calls for identical requests.
-   **Token Usage Tracking:** Estimated token counts for each LLM interaction.
-   **Cost Estimation:** Calculated cost per LLM call (displayed in logs/results).
-   Choose cost-effective models like `gpt-3.5-turbo` or `claude-3-haiku` for development and large-scale processing.

## Evaluation Metrics

The pipeline includes a comprehensive evaluation system to assess performance:

-   **Completeness**: Success rate of completed pipeline stages.
-   **Consistency**: Internal agreement and logical soundness of extracted data (especially if a validation chain is used).
-   **Token Efficiency**: Ratio of input tokens to total tokens used by the LLMs, and associated costs.
-   **Processing Time**: Total time taken for document processing and LLM extractions.
-   **Content Quality**: Accuracy and relevance of extracted information compared to ground truth (if available).
-   **Cost Efficiency**: Balance between successful extractions and total expenditure.

## Advanced Features

### Custom Prompt Management (Chain Builder)
The Streamlit "Chains" tab allows users to define and save custom extraction prompts. These custom prompts can then be selected from the "Custom Extraction" option in the "Documents" tab for tailored information retrieval.

### Batch Processing
Process multiple documents efficiently:

```python
# Example: Using DocumentProcessor for batch handling
from pipeline.document_processor import DocumentProcessor

processor = DocumentProcessor()
# Assuming 'document_paths' is a list of file paths
results = processor.batch_process_documents(document_paths)
# You would then iterate through these results to run ExtractionPipeline for each
````

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
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

1.  **API Key Not Found / Insufficient Quota**

      - Check your `.env` file exists and contains your API key.
      - Verify the key is correctly formatted.
      - **If you receive "Quota Exceeded" (Error 429) or similar billing errors:** Check your API provider's billing dashboard and ensure sufficient credits. You can also switch to an alternate LLM provider (e.g., from OpenAI to Anthropic) in the Streamlit sidebar if you have an active key for them.

2.  **Document Processing Errors**

      - Ensure the document is not corrupted.
      - Check file size limits (default 10MB) in `config.py`.
      - Verify the file format is supported (`.pdf`, `.docx`, `.txt`, `.md`).

3.  **Memory Issues**

      - Reduce `CHUNK_SIZE` in `config.py` for large documents.
      - Ensure document caching is enabled.
      - Process documents individually if batch processing causes issues.

4.  **Slow Processing**

      - Use faster, more cost-effective models (e.g., `gpt-3.5-turbo` or `claude-3-haiku`).
      - Reduce `CHUNK_OVERLAP` in `config.py`.
      - Consider if parallel processing is needed (currently not explicitly implemented in `_run_pipeline`).

### Debug Mode

```bash
# Run with debug logging
streamlit run app.py --logger.level=debug

# Check logs
tail -f data/logs/pipeline_*.log
```

## Contributing

1.  Fork the repository
2.  Create a feature branch
3.  Make your changes
4.  Add tests for new functionality
5.  Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

  - Check the troubleshooting section.
  - Review the application logs (`data/logs/pipeline_*.log`).
  - Create an issue in the repository.

## Changelog

### Version 1.0.0

  - Initial release
  - Multi-LLM support (OpenAI, Anthropic)
  - Configurable extraction tasks and custom prompt builder
  - Streamlit UI for interactive use
  - Comprehensive evaluation system
  - Export functionality for results
  - Intelligent caching for cost and performance optimization
  - Robust API error handling and retry mechanisms

## Future Enhancements

  - [ ] Integration with more LLM providers (e.g., self-hosted/open-source models like Hugging Face).
  - [ ] Advanced document parsing (e.g., tables, images, charts).
  - [ ] Multi-language support for documents and extractions.
  - [ ] REST API endpoint for programmatic access.
  - [ ] Enhanced visualization features for extracted data and evaluation metrics.
  - [ ] Collaborative workspace features for team usage.
  - [ ] Integration with document management systems (DMS).

<!-- end list -->