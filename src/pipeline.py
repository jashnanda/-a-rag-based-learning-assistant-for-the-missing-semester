import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_anthropic import ChatAnthropic
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import List, Any

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful teaching assistant for The Missing Semester of Your CS Education, an MIT course on practical computing tools.

You will be given a set of course material excerpts (retrieved via RAG) and a student question.
Your job is to answer the question using ONLY the provided excerpts.

Rules:
- Base your answer strictly on the excerpts. Do not add outside knowledge.
- If the answer is not covered in the excerpts, say so clearly.
- At the end of every answer, include a "Sources:" line listing the excerpt filenames you actually used.
- If you used no excerpts (e.g. the question is out of scope), write "Sources: none"."""

# Few-shot examples shown as prior turns in the conversation.
# Each example teaches the model the expected input/output format,
# especially the "Sources:" citation pattern.
FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": """Course material excerpts:
[Excerpt 1 — 2020-01-22-version-control.md]
The `git log` command shows the commit history. Each commit has a hash, author, date, and message. You can use `git log --oneline` for a compact view.

[Excerpt 2 — 2020-01-22-version-control.md]
`git diff <commit>` shows the changes introduced by a commit relative to your working tree. `git show <commit>` displays the full patch for that commit.

Student question: How can I see what changed in a specific git commit?""",
    },
    {
        "role": "assistant",
        "content": """To see what changed in a specific git commit, use:

```
git show <commit>
```

This displays the full patch (diff) for that commit along with its metadata (author, date, message).

If you just want to compare a commit against your current working tree:

```
git diff <commit>
```

To find the commit hash in the first place, use `git log` or `git log --oneline` for a compact list.

Sources: 2020-01-22-version-control.md""",
    },
    {
        "role": "user",
        "content": """Course material excerpts:
[Excerpt 1 — 2020-01-21-shell-tools.md]
The `find` command recursively searches for files. Example: `find . -name "*.py"` finds all Python files under the current directory. You can also filter by type with `-type f` (files) or `-type d` (directories).

[Excerpt 2 — 2020-01-21-shell-tools.md]
`grep -r "pattern" .` searches recursively through files for a pattern. The `-l` flag prints only filenames that match.

Student question: What is the difference between grep and find?""",
    },
    {
        "role": "assistant",
        "content": """Based on the excerpts:

- **`find`** searches for *files and directories* by name, type, or other attributes. Example:
  ```
  find . -name "*.py"        # find all Python files
  find . -type d             # find all directories
  ```

- **`grep`** searches for a *text pattern inside files*. Example:
  ```
  grep -r "pattern" .        # search recursively through file contents
  grep -rl "pattern" .       # print only filenames that match
  ```

In short: use `find` to locate files, use `grep` to search within file contents.

Sources: 2020-01-21-shell-tools.md""",
    },
]


def compress_chunks(docs: list[Document], query: str) -> list[Document]:
    # filter out irrelevant sentences in each chunk, use cheapest claude model since its good enoughh
    llmModel = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])
    compressor = LLMChainExtractor.from_llm(llmModel)
    compressed = compressor.compress_documents(docs, query)

    #if compressor accidentally strips too much/everytging, we just return docs as normal
    if compressed:
        return compressed
    else:
        return docs

class CustomHybridRetriever(BaseRetriever):
    chunks: List[Any] = Field(default_factory=list)
    raw_texts: List[str] = Field(default_factory=list)
    vectorstore: Any = None
    top_k: int = 5
    rerank: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from retriever import hybrid_retrieve
        return hybrid_retrieve(query, self.chunks, self.raw_texts, self.vectorstore, top_k=self.top_k, rerank_results=self.rerank)

def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[Excerpt {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def answer(
    query: str,
    chunks: list[Document],
    raw_texts: list[str],
    vectorstore: Chroma,
    top_k: int = 5,
    compress: bool = False,
    multiquery: bool = True,
    rerank: bool = True,
) -> dict:
    if multiquery:
        llmModel = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])
        base_retriever = CustomHybridRetriever(
            chunks=chunks,
            raw_texts=raw_texts,
            vectorstore=vectorstore,
            top_k=top_k,
            rerank=rerank,
        )
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llmModel
        )
        retrieved = mq_retriever.invoke(query)
    else:
        from retriever import hybrid_retrieve
        retrieved = hybrid_retrieve(query, chunks, raw_texts, vectorstore, top_k=top_k, rerank_results=rerank)

    if compress:
        retrieved = compress_chunks(retrieved, query)
    context = format_context(retrieved)

    user_message = f"""Course material excerpts:
{context}

Student question: {query}"""

    messages = FEW_SHOT_TURNS + [{"role": "user", "content": user_message}]

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return {
        "question": query,
        "answer": response.content[0].text,
        "sources": [Path(doc.metadata.get("source", "unknown")).name for doc in retrieved],
        "retrieved_docs": retrieved,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from embed_and_store import load_vectorstore, load_bm25_data

    vectorstore = load_vectorstore()
    chunks, raw_texts = load_bm25_data()

    query = "How do I use git to undo a commit?"
    result = answer(query, chunks, raw_texts, vectorstore)

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources: {result['sources']}")
