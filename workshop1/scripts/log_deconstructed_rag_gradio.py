# ### Objective:
# - Using the RAG system from HW1, add logging capabilities to the system in orde to create a minimum viable evaluation setup.
# - Log interaction into a json file!

import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from rich import print
from torch.nn import functional as F

MODEL_CONFIG = {
        "model_name": "gpt-4o-mini",
        "model_provider": "openai",
        "model_parameters": {
            "max_tokens": 1000,
            "top_p": 1,
            "temperature": 0.3,
        }
    }

RESTRICTIONS = """
Don't hallucinate
Don't provide information that is not present in the context. Apologize and request more information if the context is not helpful.
Cross-question the user to get more information if the context is not helpful.
"""

PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions based on the provided context within the restrictions permitted
Context:
{context}

Question:
{query}

Restrictions:
{restrictions}

Answer:
"""




# Replace the database reading code with JSON reading
def read_interactions_log(log_dir: Path) -> pd.DataFrame:
    interactions_file = log_dir / "interactions.json"
    if interactions_file.exists():
        with open(interactions_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.DataFrame()


## 1. Document and Nodes
class Document:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}

class Node:
    def __init__(
            self, 
            text: str, 
            metadata: Optional[Dict[str, Any]] = None,
            node_id: Optional[str] = None):
        self.text = text
        self.metadata = metadata or {}
        self.node_id = node_id or f"node_{id(self)}" ## Simple Unique ID if not provided

    def __repr__(self):
        return f"Node(id={self.node_id}, text={self.text[:50]}, metadata={self.metadata})"
    

class Response:
    def __init__(self, response: str, prompt: str) -> None:
        self.response = response
        self.prompt = prompt

    def __str__(self) -> str:
        return self.response
    
## 3. For Indexing      
class SimpleNodeParser:
    def __init__(self, chunk_size: int = 4096, chunk_overlap: int= 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overalap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i+self.chunk_size]
            if chunk:
                chunks.append(chunk)

        return chunks

    def get_nodes_from_documents(self, documents: List[Document]) -> List[Node]:
        """Convert documents to nodes by splitting text into chunks"""
        nodes = []
        for doc in documents:
            text_chunks = self.split_text(doc.text)
            for i, text_chunk in enumerate(text_chunks):
                ## Copy metadata and add chunk info:
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),

                })
                nodes.append(Node(
                    text=text_chunk, metadata=metadata))
                
        return nodes
    
class SimpleDirectoryReader:
    """Read all text files from a directory and return a list of documents."""
    def __init__(self, directory_path: str) -> None:
        self.directory_path = directory_path

    def load_data(self) -> List['Document']:
        """ Load all text files from the directory. """
        documents = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.directory_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                documents.append(Document(text, metadata={"source": filename}))
        return documents



## Need to create a base embedding class that extensible for different embedding models: Cohere and HuggingFace.
class BaseEmbedding:
    def get_embedding(self, text: List[str]) -> List[float]:
        """Get embedding from a single text file using API."""
        raise NotImplementedError("Subclasses must implement this method.")

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "text-embedding-ada-002") -> None:
        self.model_name = model_name

    def get_embedding(self, text: str ) -> List[float]:
        """Get embedding from a single text file using OpenAI API."""
        response = openai.embeddings.create(
            model=self.model_name,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding
    
    def get_embeddings(self, texts:List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        return [self.get_embedding(text) for text in texts]
    
class LLMResponseSynthesizer:
    def __init__(self, restrictions: str = RESTRICTIONS, prompt_template: str = PROMPT_TEMPLATE, model_config: Dict[str, Any] = MODEL_CONFIG) -> None:
        self.client = openai.OpenAI() ## Need to update this to support different LLM providers.
        self.model_config = model_config

        self.restrictions = restrictions
        self.prompt_template = prompt_template

    def synthesize(self, query: str, nodes: List[Node]) -> Response:
        """Synthesize a response from the context and query using the initialized LLM.        
        """

        ## Build context from nodes:
        context = "\n\n".join([f"Document chunk: {node.text}" for node in nodes])

        ## Build the prompt with context and query:
        prompt = self.prompt_template.format(context=context, query=query, restrictions=self.restrictions)
        # print(prompt)

        ## Call OpenAI API:
        response = self.client.chat.completions.create(
            model=self.model_config['model_name'],
            max_tokens=self.model_config['model_parameters']['max_tokens'],
            top_p=self.model_config['model_parameters']['top_p'],
            temperature=self.model_config['model_parameters']['temperature'],
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )

        return Response(response=response, prompt=prompt) # type: ignore


class SimpleVectorStore:
    """Simple Vector Store: Functionalities to add nodes, retrieve nodes based on similarity search (top_k using cosine similarity)"""
    def __init__(self) -> None:
        self.embeddings = []
        self.node_ids = []
        self.node_dict = {} ## Store actual node objects by ID

    def add_notes(self, nodes: List[Node], embeddings: List[List[float]]) -> None:
        for node, embedding in zip(nodes, embeddings):
            self.embeddings.append(embedding)
            self.node_ids.append(node.node_id)
            self.node_dict[node.node_id] = node

    def similarity_search(self, query_embedding: List[float], top_k: int = 2) -> List[Node]:
        """Find `top_k` most similar nodes to the query embedding using cosine similarity."""

        if not self.embeddings:
            logging.warning("No embeddings in the vector store.")
            return []
        
        ## Convert lists to tensor 
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        embeddings_tensor = torch.tensor(self.embeddings, dtype=torch.float32)

        ##Normalize the query embedding and c
        query_tensor = F.normalize(query_tensor, p=2, dim=0)
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

        ## Compute cosine similarities:
        similarities = torch.matmul(query_tensor, embeddings_tensor.T)

        ##Get top_k indices:
        top_indices = torch.argsort(similarities, descending=True)[:top_k].tolist()

        ## Return the nodes corresponding to the top_k indices:
        return [self.node_dict[self.node_ids[idx]] for idx in top_indices]
    
## Query Engine:
class QueryEngine:
    def __init__(
            self, 
            vector_store: SimpleVectorStore, 
            response_synthesizer: LLMResponseSynthesizer, 
            similarity_topk: int = 2,
            ) -> None:
        self.vector_store = vector_store
        self.response_synthesizer = response_synthesizer
        self.embedding_service = OpenAIEmbedding()
        self.similarity_topk = similarity_topk

    def query(self, query: str) -> Response:
        """Execute the query and return the response."""

        ## Get query embedding:
        query_embedding = self.embedding_service.get_embedding(query) # type: ignore ## Single statement so we use `get_embedding`

        ## Retrieve the relevant nodes using similarity search:
        retrieved_nodes = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.similarity_topk,
        )

        ## Generate response
        response = self.response_synthesizer.synthesize(query=query, nodes=retrieved_nodes)

        return response
    
## Vector Store Index:
class VectorStoreIndex:
    """Vector Store Index: Manage nodes, embeddings, and vector store."""
    def __init__(self, nodes: List[Node], vector_store: SimpleVectorStore, similarity_topk: int = 2) -> None:
        self.nodes = nodes
        self.vector_store = vector_store
        self.similarity_topk = similarity_topk

    @classmethod
    def from_documents(
        cls, 
        documents: List[Document],
        embedding_service: OpenAIEmbedding,
        node_parser=None) -> 'VectorStoreIndex':

        """ Create index from documents"""
        ## Initialize the embedding service:
        embedding_service = embedding_service or OpenAIEmbedding()
        node_parser = node_parser or SimpleNodeParser()

        ## Create nodes from documents:
        nodes = node_parser.get_nodes_from_documents(documents=documents)

        ## Get embeddings for all nodes:
        texts = [node.text for node in nodes]
        embeddings = embedding_service.get_embeddings(texts=texts)

        ## Create and populate vector store:
        vector_store = SimpleVectorStore()
        vector_store.add_notes(nodes=nodes, embeddings=embeddings)

        return cls(nodes=nodes, vector_store=vector_store)
    
    def as_query_engine(self, response_synthesizer: LLMResponseSynthesizer, similarity_topk: int = 2) -> QueryEngine:
        """Create a query engine from this index"""
        response_synthesizer = response_synthesizer or LLMResponseSynthesizer() ## Default to a simple response synthesizer
        return QueryEngine(
            vector_store=self.vector_store,
            response_synthesizer=response_synthesizer,
            similarity_topk=similarity_topk,
        )

## Configure logging to print to console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ### Modified RAG-System with logging capabilities


class RAGInteractionLogger:
    """Handles logging of RAG system interactions to JSON files"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.interactions_file = self.log_dir / "interactions.json"
        self.system_logs_file = self.log_dir / "system_logs.json"
        self._init_log_files()

    def _init_log_files(self) -> None:
        """Initialize JSON log files if they don't exist"""
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize interactions file
        if not self.interactions_file.exists():
            self.interactions_file.write_text('[]')
            logger.info(f"Created interactions log file at {self.interactions_file}")
        
        # Initialize system logs file
        if not self.system_logs_file.exists():
            self.system_logs_file.write_text('[]')
            logger.info(f"Created system logs file at {self.system_logs_file}")
    
    def _read_json_file(self, file_path: Path) -> List[Dict]:
        """Helper method to read JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    
    def _write_json_file(self, file_path: Path, data: List[Dict]) -> None:
        """Helper method to write JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_system_event(
            self, 
            level: str,
            message: str,
            module: str,
            function: str,
            traceback: str,
            llm_config: Optional[Dict[str, Any]] = None,
            ) -> None:
        """Log system events to JSON file"""
        log_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'module': module,
            'function': function,
            'traceback': traceback,
            'llm_config': llm_config or {}
        }
        
        logs = self._read_json_file(self.system_logs_file)
        logs.append(log_entry)
        self._write_json_file(self.system_logs_file, logs)
    
    def log_interaction(self, 
                       query: str, 
                       response: str, 
                       source_document: str,
                       system_prompt: str,
                       model_name: str,
                       model_type: str,
                       model_parameters: Dict[str, Any],
                       retrieved_context: List[str],
                       metadata: Optional[Dict[str, Any]] = None,
                       processing_time: float = 0.0,
                       token_usage: Optional[Dict[str, int]] = None) -> None:
        """Log an interaction to JSON file"""
        interaction = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'source_document': source_document,
            'system_prompt': system_prompt,
            'model_name': model_name,
            'model_type': model_type,
            'model_parameters': model_parameters,
            'retrieved_context': retrieved_context,
            'processing_time': processing_time,
            'token_usage': token_usage or {},
            'metadata': metadata or {}
        }
        
        interactions = self._read_json_file(self.interactions_file)
        interactions.append(interaction)
        self._write_json_file(self.interactions_file, interactions)



class RAGwithLogging:
    def __init__(self, data_dir: str, model_config: Dict[str, Any], log_dir: str = "logs", similarity_topk: int = 2, restrictions: str = RESTRICTIONS, prompt_template: str = PROMPT_TEMPLATE):
        self.data_dir = data_dir
        self.similarity_topk = similarity_topk
        self.model_config = model_config
        self.restrictions = restrictions
        self.prompt_template = prompt_template
        self.logger = RAGInteractionLogger(log_dir=log_dir)
        self.documents = self._load_documents()
        self.vector_index = self.create_vector_index()
        self.query_engine = self._create_query_engine() ## Fundctions not meant to be called directly.

    def _load_documents(self) -> List[Document]:
        """
        Load documents from directory and log the process.
        """
        try:
            documents = []
            for filename in os.listdir(self.data_dir)[:3]:
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.data_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    documents.append(Document(text, metadata={"source": filename}))
            return documents
        except Exception as e: 
            self.logger.log_system_event(
                level="ERROR",
                message=f"Error loading documents: {str(e)}",
                module=__name__,
                function="_load_documents",
                traceback=traceback.format_exc()
            )
            raise Exception(f"Error loading documents: {str(e)}")
        
    def create_vector_index(self) -> VectorStoreIndex:
        """Create vector index from documents and log the process"""
        try:
            node_parser = SimpleNodeParser()
            embedding_service = OpenAIEmbedding()
            nodes = node_parser.get_nodes_from_documents(self.documents)
            texts = [node.text for node in nodes]
            embeddings = embedding_service.get_embeddings(texts)
            
            vector_store = SimpleVectorStore()
            vector_store.add_notes(nodes=nodes, embeddings=embeddings)

            return VectorStoreIndex(nodes=nodes, vector_store=vector_store, similarity_topk=self.similarity_topk)
        except Exception as e:
            self.logger.log_system_event(
                level="ERROR",
                message=f"Error creating vector index: {str(e)}",
                module=__name__,
                function="_create_vector_index",
                traceback=traceback.format_exc()
            )   
            raise Exception(f"Error creating vector index: {str(e)}")
        
    def _create_query_engine(self) -> QueryEngine:
        """Create query engine from vector index and log the process"""
        try:
            response_synthesizer = LLMResponseSynthesizer(model_config=self.model_config, restrictions=self.restrictions, prompt_template=self.prompt_template)
            return self.vector_index.as_query_engine(response_synthesizer=response_synthesizer, similarity_topk=self.similarity_topk)
        except Exception as e:
            self.logger.log_system_event(
                level="ERROR",
                message=f"Error creating query engine: {str(e)}",
                module=__name__,
                function="_create_query_engine",
                traceback=traceback.format_exc()
            )
            raise Exception(f"Error creating query engine: {str(e)}")
        
    def query(self, query: str, model_config: Dict[str, Any]) -> Union[str, None]:
        """Execute query with enhanced logging"""
        start_time = datetime.now()
        try:
            # Get the response and context
            response = self.query_engine.query(query)
            query_embedding = self.query_engine.embedding_service.get_embedding(query)
            retrieved_nodes = self.query_engine.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=self.query_engine.similarity_topk
            )
            retrieved_context = [node.text for node in retrieved_nodes]
            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.log_interaction(
                query=query,
                response=str(response.response),
                source_document=self.documents[0].metadata.get("source", "unknown"),
                system_prompt=response.prompt,
                model_name=self.model_config['model_name'],
                model_type=self.model_config['model_provider'],
                model_parameters=self.model_config['model_parameters'],
                retrieved_context=retrieved_context,
                metadata={"processing_time": processing_time},
                processing_time=processing_time
            )
            return response.response

            

        except Exception as e:
            self.logger.log_system_event(
                level="ERROR",
                message=f"Error executing query: {str(e)}",
                module=__name__,
                function="_query",
                traceback=traceback.format_exc()
            )
            


def create_gradio_interface(rag_system: RAGwithLogging) -> gr.Interface:
    """Create a Gradio interface for the RAG system"""
    
    def process_query(query: str, top_k: int) -> str:
        """Process query through RAG system and return formatted response"""
        try:
            # Update similarity_topk based on user input
            rag_system.query_engine.similarity_topk = top_k
            
            # Process query
            response = rag_system.query(query=query, model_config=MODEL_CONFIG)
            
            if response:
                # Get the most recent interaction from logs to show token usage
                with open(rag_system.logger.interactions_file, 'r') as f:
                    logs = json.load(f)
                    latest_interaction = logs[-1] if logs else {}
                usage_response = response.usage ## type: ignore
                # Format the response with metadata
                chat_response = response.choices[0].message.content ## type: ignore
                formatted_response = f"""
---------------------------------------------------------------------------
### Response:
{chat_response}

### Metadata:
- Processing Time: {latest_interaction.get('processing_time', 'N/A'):.2f} seconds
- Token Usage:
  - Prompt Tokens: {usage_response.prompt_tokens or 'N/A'}
  - Completion Tokens: {usage_response.completion_tokens or 'N/A'}
  - Total Tokens: {usage_response.total_tokens or 'N/A'}
---------------------------------------------------------------------------
""" 
                return formatted_response
            return "No response generated."
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    # Create the Gradio interface
    interface = gr.Interface(
        fn=process_query,
        inputs=[
            gr.Textbox(
                label="Enter your question",
                placeholder="What would you like to know?",
                lines=3
            ),
            gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of similar documents to retrieve (top-k)"
            )
        ],
        outputs=gr.Markdown(label="Response"),
        title="RAG System with Logging",
        description="""
        This is a RAG (Retrieval-Augmented Generation) system that can answer questions based on the provided documents.
        The system logs all interactions and provides detailed token usage information.
        """,
        examples=[
            ["What is the main topic of the documents?", 3],
            ["Can you summarize the key points?", 5],
        ],
        theme="soft"
    )
    
    return interface

def main():
    # Load environment variables
    load_dotenv(dotenv_path="../../project_secrets.env")
    load_dotenv(dotenv_path="../ai_sdlc_secrets.env")

    root_dir = Path(os.environ.get("ROOT_DIR", "../.."))
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Configure data directory
    data_dir = root_dir  # Update this path as needed
    
    try:
        # Initialize RAG system
        print(f"Initializing RAG system with data from: {data_dir}")
        rag_system = RAGwithLogging(
            data_dir=str(data_dir),
            model_config=MODEL_CONFIG,
            log_dir=str(log_dir),
            similarity_topk=3
        )
        
        # Create and launch Gradio interface
        interface = create_gradio_interface(rag_system)
        interface.launch(
            server_name="0.0.0.0",  # Makes the app accessible from other machines
            server_port=7860,        # Default Gradio port
            share=True              # Creates a public URL
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()








