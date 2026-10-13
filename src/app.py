import sys
from pathlib import Path
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from src.config import get_llm, get_embeddings

MAX_REGEN, MAX_REWRITE = 2, 1
SAFE_REFUSAL = "I can't answer that reliably from the provided documents."

class State(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    regens: int
    rewrites: int
    trace: List[str]

class YesNo(BaseModel):
    ok: bool = Field(description="Binary decision")

llm, retriever = None, None

def _grade(template, **kw):
    return (ChatPromptTemplate.from_template(template) | llm.with_structured_output(YesNo)).invoke(kw).ok

def retrieve(s):
    return {"documents": retriever.invoke(s["question"]), "trace": s["trace"] + ["retrieve"]}

def grade_docs(s):
    kept = [d for d in s["documents"] if _grade(
        "Question: {q}\nPassage: {p}\nIs the passage relevant?", q=s["question"], p=d.page_content)]
    return {"documents": kept, "trace": s["trace"] + [f"grade_docs kept={len(kept)}"]}

def generate(s):
    strict = "Use ONLY the context. Every statement must be directly supported. " if s["regens"] else ""
    ctx = "\n\n".join(d.page_content for d in s["documents"])
    p = ChatPromptTemplate.from_template(
        strict + "Answer ONLY from the context; if unsure say you don't know.\n\n{ctx}\n\nQ: {q}")
    return {"generation": (p | llm | StrOutputParser()).invoke({"ctx": ctx, "q": s["original_question"]}),
            "trace": s["trace"] + ["generate"]}

def check_grounded(s):
    ctx = "\n\n".join(d.page_content for d in s["documents"])
    ok = _grade("Context:\n{c}\n\nAnswer: {a}\n\nIs EVERY claim in the answer supported by the context?",
                c=ctx, a=s["generation"])
    return {"trace": s["trace"] + [f"grounded={ok}"], "regens": s["regens"] + (0 if ok else 1),
            "generation": s["generation"] if ok else ""}

def check_useful(s):
    ok = _grade("Question: {q}\nAnswer: {a}\nDoes the answer actually resolve the question?",
                q=s["original_question"], a=s["generation"])
    return {"trace": s["trace"] + [f"useful={ok}"], "generation": s["generation"] if ok else ""}

def rewrite(s):
    q = (ChatPromptTemplate.from_template("Improve this as a search query. Return only the query.\n{q}")
         | llm | StrOutputParser()).invoke({"q": s["question"]})
    return {"question": q, "rewrites": s["rewrites"] + 1, "trace": s["trace"] + ["rewrite"]}

def refuse(s):
    return {"generation": SAFE_REFUSAL, "trace": s["trace"] + ["refuse"]}

def after_docs(s):
    if s["documents"]: return "generate"
    return "rewrite" if s["rewrites"] < MAX_REWRITE else "refuse"

def after_grounded(s):
    if s["generation"]: return "check_useful"
    return "generate" if s["regens"] <= MAX_REGEN else "refuse"

def after_useful(s):
    if s["generation"]: return END
    return "rewrite" if s["rewrites"] < MAX_REWRITE else "refuse"

def build():
    g = StateGraph(State)
    for n, f in [("retrieve", retrieve), ("grade_docs", grade_docs), ("generate", generate),
                 ("check_grounded", check_grounded), ("check_useful", check_useful),
                 ("rewrite", rewrite), ("refuse", refuse)]:
        g.add_node(n, f)
    g.add_edge(START, "retrieve"); g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", after_docs)
    g.add_edge("generate", "check_grounded")
    g.add_conditional_edges("check_grounded", after_grounded)
    g.add_conditional_edges("check_useful", after_useful)
    g.add_edge("rewrite", "retrieve"); g.add_edge("refuse", END)
    return g.compile()

if __name__ == "__main__":
    llm = get_llm()
    docs = []
    for f in Path("data").glob("*.txt"):
        docs += TextLoader(str(f), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)
    retriever = FAISS.from_documents(chunks, get_embeddings()).as_retriever(search_kwargs={"k": 4})
    q = " ".join(sys.argv[1:]) or "How long do I have to submit receipts?"
    out = build().invoke({"question": q, "original_question": q, "documents": [], "generation": "",
                          "regens": 0, "rewrites": 0, "trace": []})
    print("TRACE :", " -> ".join(out["trace"]))
    print("ANSWER:", out["generation"])
