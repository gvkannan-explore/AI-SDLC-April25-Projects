### Objective: Use Llama-Index to create a RAG system which intakes a single document - LinkedIn profile and answers questions about it using LLMs.

## Extensions:
# 1. Use OCR to recognize the text and convert it to a string and then re-run the pipeline.
# 2. Create embeddings from other profiles and recommend similar profiles.

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Union

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from rich import print


@dataclass
class GeminiConfig:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class ClaudeConfig:
    model: str = "claude-3-7-sonnet-latest"
    temperature: float = 0.1
    max_tokens: int = 512


class RAG_Pipeline:
    """
    A simple RAG pipeline that uses the LlamaIndex library to create a vector store index from documents in a directory and allows querying from index using configured LLM.
    """

    def __init__(
        self,
        data_dir: Union[str, PosixPath],
        llm_provider: str = "GoogleGenAI",
        llm_config: GeminiConfig = GeminiConfig(),
    ):
        self.data_dir = data_dir
        self.documents = SimpleDirectoryReader(
            input_dir=self.data_dir
        ).load_data()  ## Similar to pd.read_csv()
        self.index = VectorStoreIndex.from_documents(
            self.documents,
        )  ## Uses open-ai-embeddings so fails without the API key.

        if llm_provider == "GoogleGenAI":
            self.llm_cfg = llm_config
            self.llm = GoogleGenAI(
                model=self.llm_cfg.model_name,
                temperature=self.llm_cfg.temperature,
                max_tokens=self.llm_cfg.max_tokens,
            )

        elif llm_provider == "Claude":
            self.llm_cfg = llm_config
            self.llm = Anthropic(
                model=self.llm_cfg.model,
                temperature=self.llm_cfg.temperature,
                max_tokens=self.llm_cfg.max_tokens,
            )
        else:
            raise Exception(
                f"Invalid LLM provided: {llm_provider}. Supported LLMs are: `GoogleGenAI`"
            )

        self.query_engine = self.index.as_query_engine(llm=self.llm)

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
        "--llm-provider",
        type=str,
        default="GoogleGenAI",
        choices=["GoogleGenAI", "Claude"],
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask reg. the data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(dotenv_path=args.api_secret)
    if args.llm_provider == "GoogleGenAI":
        llm_cfg = GeminiConfig()
    elif args.llm_provider == "Claude":
        llm_cfg = ClaudeConfig()
    else:
        raise Exception(
            f"Invalid LLM provider: {args.llm_provider}. Supported LLMs are: `GoogleGenAI`, `Claude`"
        )

    pipe = RAG_Pipeline(
        data_dir=args.data_dir,
        llm_provider=args.llm_provider,
        llm_config=llm_cfg,
    )

    print("=" * 50)
    print(f"Query: {args.query}")
    print(f"Response: {pipe.query(args.query)}")
    print("=" * 50)