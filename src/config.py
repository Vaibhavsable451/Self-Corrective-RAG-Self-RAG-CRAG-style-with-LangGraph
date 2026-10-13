import os
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)

def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL)
