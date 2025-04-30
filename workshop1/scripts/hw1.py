### Objective: Without using Llama-Index to create a simple RAG system which intakes a single document - LinkedIn profile and answers questions about it.
import os
import gradio as gr
import fitz ## PyMuPDF
import sqlite3
from datetime import datetime
import uuid
from tqdm.notebook import tqdm
import traceback
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Union
import logging
from rich import print
import openai
import torch
from torch.nn import functional as F
import argparse

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
    
## 3. For Indexing      
class SimpleNodeParser:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int= 20) -> None:
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

    def get_embedding(self, text: str) -> List[float]:
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
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name
        self.client = openai.OpenAI()

        self.restrictions = """
        Don't hallucinate
        """
        self.prompt_template = """
        You are a helpful assistant that can answer questions based on the provided context within the restrictions permitted
        Context:
        {context}

        Question:
        {query}

        Restrictions:
        {restrictions}

        Answer:
        """

    def synthesize(self, query: Union[str, List[str]], nodes: List[Node]) -> str:
        """Synthesize a response from the context and query using the initialized LLM.        
        """

        ## Build context from nodes:
        context = "\n\n".join([f"Document chunk: {node.text}" for node in nodes])

        ## Build the prompt with context and query:
        prompt = self.prompt_template.format(context=context, query=query, restrictions=self.restrictions)

        ## Call OpenAI API:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            
        )
        if response.choices[0].message.content is None:
            return "No response from the model."
        else:
            return response.choices[0].message.content
    
class Response:
    def __init__(self, response: str,) -> None:
        self.response = response

    def __str__(self) -> str:
        return self.response

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
        query_embedding = self.embedding_service.get_embedding(query) ## Singe statement so we use `get_embedding`

        ## Retrieve the relevant nodes using similarity search:
        retrieved_nodes = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.similarity_topk,
        )

        ## Generate response
        response = self.response_synthesizer.synthesize(query=query, nodes=retrieved_nodes)

        return Response(response=response)
    
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
        node_parser=None):

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
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG system.")
    parser.add_argument("--query", type=str, help="The query to search the RAG system with.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    load_dotenv(dotenv_path="../../project_secrets.env")
    load_dotenv(dotenv_path="../../../ai_sdlc_secrets.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    documents = SimpleDirectoryReader("../apps/data").load_data() ## TO DO: Update this to only add files that are relevant to the query


    ## Unleash the RAG!
    vector_index = VectorStoreIndex.from_documents(documents=documents, embedding_service=OpenAIEmbedding())
    query_engine = vector_index.as_query_engine(response_synthesizer=LLMResponseSynthesizer(), similarity_topk=2)
    response = query_engine.query(query=args.query)

    print(response)