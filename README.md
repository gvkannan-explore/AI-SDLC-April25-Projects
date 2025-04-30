# AI-SDLC-April25-Projects
"Homework" and projects based on the content covered in `Building LLM Applications for Data Scientists & Software Engineers`

# Homework - RAG System with Logging Capabilities

This project implements a Retrieval-Augmented Generation (RAG) system with comprehensive logging capabilities. It includes both a CLI and Gradio web interface version.

## Features

- Document loading and chunking
- Vector embeddings using OpenAI's embedding model
- Similarity search for relevant context retrieval
- Response generation using OpenAI's GPT models
- Comprehensive logging of interactions and system events
- Both CLI and web interface options

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
uv venv --python 3.12
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
uv pip install -r requirements.txt
```

4. Set up environment variables:
Update a `.env` file in the project root with and with the OpenAI API

5. Next steps:
    a. Add relevant chunks instead of the entire prompt to logging that was used for answers so that we can evaluate if the chunk retrieved was appropriate.
    b. Support Gemini and Claude calls


