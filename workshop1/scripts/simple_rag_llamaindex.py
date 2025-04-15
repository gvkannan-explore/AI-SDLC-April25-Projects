### Objective: Use Llama-Index to create a simple RAG system which intakes documents from a directory - LinkedIn profiles and answers questions about it.

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Union
from dotenv import load_dotenv
from pathlib import Path
import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.anthropic import Anthropic
from dataclasses import dataclass
import logging
import sys
from rich import print


## Definition/Usage:
# VectorStoreIndex - This is a vector store index that allows you to create an index from a set of documents. Allows us to index the relevant documents and then query them.
# SimpleDirectoryReader - This is a simple directory reader that allows you to read documents from a directory into LlamaIndex. 

class SimpleRAG:
    def __init__(self, data_dir: Union[str, PosixPath]):
        self.data_dir = data_dir
        self.documents = SimpleDirectoryReader(input_dir=self.data_dir).load_data() ## Similar to pd.read_csv()
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine()

    def query(self, query: str) -> str:
        response = self.query_engine.query(query)
        return response
    

def parse_args():
    parser = ArgumentParser(
        description="RAG Pipeline using LlamaIndex and configured LLM.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing documents.",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        required=True,
        help="Environment file that contains the API keys.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask reg. the data",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.api_secret)
    pipe = SimpleRAG(data_dir=args.data_dir)

    print("=" * 50)
    print(f"Query: {args.query}")
    print(f"Response: {pipe.query(args.query)}")
    print("=" * 50)