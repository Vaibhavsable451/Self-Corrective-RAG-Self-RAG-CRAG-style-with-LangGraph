# 05 - Self-Corrective RAG (Self-RAG / CRAG style) with LangGraph

Adds **output-side hallucination control** to the retrieval loop:

```
retrieve -> grade_docs -> generate -> check_grounded --(no)--> regenerate (stricter prompt, max 2)
                                           |(yes)
                                      check_useful --(no)--> rewrite_query -> retrieve
                                           |(yes)
                                          END
```
Three LLM "graders" gate the pipeline: document relevance, answer groundedness, answer usefulness.
After the retry budget is spent it **refuses safely** instead of guessing.

```bash
pip install -r requirements.txt && cp .env.example .env
python -m src.app "How long do I have to submit expense receipts?"
```
