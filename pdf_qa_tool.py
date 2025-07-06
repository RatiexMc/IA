# -*- coding: utf-8 -*-
"""Simple PDF Analysis and Question Answering tool.

This script merges the functionality of the initial `mio.py` and
`pdf_analyzer.py` examples. It can be used in Google Colab to:

1. Convert a PDF to text.
2. Clean and tokenize the text.
3. Build a word frequency table and simple visualizations.
4. Create text embeddings and a FAISS index for semantic search.
5. Answer custom questions about the PDF using a free model from
   Hugging Face.

Both the embedding model and the QA model are free to use. They may
require a bit of time to download the first time.

Example usage (in Colab):

```python
!pip install PyPDF2 sentence-transformers faiss-cpu transformers torch
from pdf_qa_tool import PDFQATool

tool = PDFQATool()
tool.load_pdf('mi_libro.pdf')
print(tool.summary())
print(tool.answer('¿Cuál es el tema principal?'))
```
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation and special characters."""
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text by simple whitespace split."""
    return text.split()


@dataclass
class PDFQATool:
    """Utility class to process a PDF and answer questions."""

    chunk_size: int = 500
    overlap: int = 50
    embedder_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    qa_model: str = "deepset/roberta-base-squad2"
    tokens: List[str] = field(default_factory=list)
    text_chunks: List[str] = field(default_factory=list)
    index: faiss.IndexFlatL2 | None = None
    full_text: str = ""

    def __post_init__(self) -> None:
        self.embedder = SentenceTransformer(self.embedder_name)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_model,
        )

    def load_pdf(self, path: str) -> None:
        """Load a PDF file, clean it and build the index."""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
        self.full_text = "\n".join(pages)
        cleaned = clean_text(self.full_text)
        self.tokens = tokenize(cleaned)
        self.text_chunks = self._split_chunks(cleaned)
        self._create_index()

    def _split_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def _create_index(self) -> None:
        """Create FAISS index from text chunks."""
        embeddings = self.embedder.encode(self.text_chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))

    def most_common(self, n: int = 10) -> List[tuple[str, int]]:
        """Return the n most common tokens."""
        return Counter(self.tokens).most_common(n)

    def answer(self, question: str, top_k: int = 5) -> str:
        """Answer a question based on the indexed PDF."""
        if not self.index:
            return "Index not ready. Load a PDF first."
        q_emb = self.embedder.encode([question]).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)
        context = " ".join(
            self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)
        )
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]

    def summary(self) -> str:
        """Return a short summary about the processed PDF."""
        wc = len(self.tokens)
        unique = len(set(self.tokens))
        return (
            f"Palabras totales: {wc}\n"
            f"Palabras únicas: {unique}\n"
            f"Chunks: {len(self.text_chunks)}"
        )

