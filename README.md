# RAG System with FactScore Evaluation

A modular Retrieval-Augmented Generation (RAG) pipeline that combines semantic search with LLM-based text generation, featuring automated factual accuracy evaluation using FactScore methodology.

## 🎯 Objective

Develop a mini RAG system that integrates information retrieval with language model-based text generation, including factuality assessment of generated responses using FactScore metrics.

## 📋 Overview

This project implements a complete RAG pipeline consisting of:

1. **Knowledge Base**: FACOM postgraduate regulations (retrieved from [facom.ufms.br](https://facom.ufms.br))
2. **Vector Indexing**: Embedding generation and storage using ChromaDB
3. **Context Retrieval**: Semantic search (top-k with MMR) for relevant document chunks
4. **Response Generation**: LLM-based answer generation (supports OpenAI API or local Mistral model)
5. **Factuality Evaluation**: FactScore calculation to measure response consistency with retrieved documents
6. **Batch Processing**: Automated evaluation across multiple questions with metrics visualization

### Architecture Highlights

- **Modular Design**: Easy to swap components (embeddings, retrievers, LLMs)
- **Flexible Inference**: Supports both OpenAI API and local Mistral-7B-Instruct-v0.2
- **Custom Embeddings**: Uses BAAI/bge-m3 for multilingual semantic search
- **Factuality Assessment**: Implements atomic fact decomposition based on the FactScore paper ([arXiv:2305.14251](https://arxiv.org/abs/2305.14251))

## 🚀 Setup

### 1. Python Virtual Environment

Create and activate a virtual environment:

```powershell
python -m venv ./venv
```

### 2. Install Dependencies

Install required packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

**Note**: If installation fails due to large packages (e.g., `transformers`, `torch`), install dependencies in smaller batches:

```powershell
# Core dependencies
pip install python-dotenv pydantic

# OpenAI and LangChain
pip install openai langchain langchain-community langchain-chroma langchain-text-splitters

# Vector store and PDF processing
pip install chromadb PyMuPDF

# ML frameworks (large downloads)
pip install torch
pip install transformers accelerate safetensors

# Visualization
pip install matplotlib
```

### 3. Environment Variables

Create a `.env` file in the project root with your OpenAI API key (if using cloud inference):

```env
OPENAI_API_KEY=your_api_key_here
```

## 📂 Project Structure

```
trabalho2-RN/
├── core/
│   ├── embeddings.py          # Custom HuggingFace embeddings (bge-m3)
│   ├── factscore.py            # FactScore evaluator (local/API)
│   ├── load.py                 # Document loading utilities
│   ├── main.py                 # Main RAG pipeline
│   ├── prompt.py               # Prompt templates
│   └── utils.py                # Helper functions
├── data/
│   └── vectorstore_manager.py  # ChromaDB management
├── subject_persist/            # ChromaDB persistence directory
├── app.py                      # Batch evaluation runner
├── init.py                     # Vector store initialization
├── questions.json              # Test questions dataset
├── regulamento.pdf             # FACOM postgraduate regulations
└── requirements.txt            # Python dependencies
```

## 🔧 Usage

### Initialize Vector Store (Run Once)

Load the regulations PDF into ChromaDB:

```powershell
python init.py
```

**Important**: Run this only once. To update the knowledge base with additional documents, modify `init.py` to include the new file path.

### Run Evaluation

Execute the batch evaluation pipeline:

```powershell
python app.py
```

This will:
- Process all questions from `questions.json`
- Generate answers using the RAG pipeline
- Calculate FactScore metrics for each response
- Save detailed results to `questions_results.json`
- Generate a factual accuracy progression chart (`factual_accuracy_progress.png`)
- Display average factual precision across all questions

### Configuration Options

To switch between local and API inference, edit `app.py`:

```python
# Use OpenAI API (default)
run_batch(local=False)

# Use local Mistral model
run_batch(local=True)
```

## 📊 Evaluation Metrics

The system tracks the following metrics per question:

- **Total Facts**: Number of atomic facts extracted from the response
- **Supported Facts**: Facts verified against retrieved context
- **Unsupported Facts**: Facts not supported by the context
- **Factual Accuracy Score**: Percentage of supported facts

Final output includes:
- JSON file with per-question detailed results
- Average factual precision across all questions
- Visual chart showing accuracy progression

## 📄 Methodology

### Prompt Engineering

Prompts are designed based on the FactScore paper methodology:
- **System Role**: Academic assistant for FACOM at UFMS
- **Constraints**: Strict adherence to retrieved context, no external knowledge
- **Citation Requirements**: Reference specific articles and sections when possible
- **In-Context Learning**: Few-shot examples for consistent formatting

### FactScore Implementation

The factuality evaluation follows these steps:
1. **Atomic Fact Decomposition**: Break down generated responses into independent statements
2. **Fact Verification**: Check each fact against retrieved context
3. **Score Calculation**: Compute precision as `supported_facts / total_facts`

Reference: *FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation* ([arXiv:2305.14251](https://arxiv.org/abs/2305.14251))

## 🛠️ Technical Details

- **Embedding Model**: BAAI/bge-m3 (multilingual, 8192 token context)
- **Vector Store**: ChromaDB with MMR retrieval for diversity
- **Text Splitting**: Recursive character splitter (600 chars, 100 overlap)
- **LLM Options**:
  - Cloud: OpenAI Responses API (gpt-5-mini)
  - Local: Mistral-7B-Instruct-v0.2 (FP16/FP32 adaptive)

## 📝 Notes

- The regulations PDF should be placed in the project root before running `init.py`
- First-time local model usage will download ~14GB (Mistral-7B)
- ChromaDB data persists in `subject_persist/chroma_persist/`
- To reset the vector store, delete the persistence directory and re-run `init.py`

## 🎓 Academic Context

This project was developed as part of a coursework assignment focused on:
- Implementing practical RAG systems
- Understanding embedding-based retrieval
- Evaluating LLM factuality in domain-specific contexts
- Combining traditional IR with modern generative AI

## 📜 License

Educational project - FACOM/UFMS