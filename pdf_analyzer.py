# -*- coding: utf-8 -*-
"""pdf_analyzer.py (versión fusionada)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Analizador de PDF + Sistema de Preguntas & Respuestas basado en
*embeddings* y FAISS.  
— Mantiene la **estructura base** del archivo original `pdf_analyzer.py`.  
— Incorpora la **lógica y utilidades** del script simplificado enviado en el
  primer mensaje (limpieza, tokenización y resumen de vocabulario).  
— Está pensado tanto para **Google Colab** (subida de archivos vía
  `files.upload`) como para uso local desde consola (CLI inter‑activo).

┌──────────────────────────────────────────────┐
│  ☑️  Requisitos (ejecutar en Google Colab)    │
└──────────────────────────────────────────────┘
```python
!pip install PyPDF2 sentence-transformers faiss-cpu transformers torch
```
"""

from __future__ import annotations

# Librerías estándar
import re
import textwrap
from collections import Counter
from typing import List

# Librerías de terceros (   ↑ instalar con la celda anterior   )
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Las siguientes solo existen en Colab; se importa de forma segura
try:
    from google.colab import files  # type: ignore
except ModuleNotFoundError:  # uso local
    files = None  # pylint: disable=invalid-name

# ╝═════════════════════════════════════════════╝
# │ Funciones auxiliares                                      │
# ╚═════════════════════════════════════════════╚

def clean_text(text: str) -> str:
    """Normaliza el texto: minúsculas y eliminación de signos de puntuación
    no alfanuméricos (se preservan tildes y eñe)."""
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    """Segmenta el texto en una lista de tokens usando espacios en blanco."""
    return text.split()

# ╝═════════════════════════════════════════════╝
# │  Clase principal                                          │
# ╚═════════════════════════════════════════════╚

class PDFAnalyzer:
    """Analizador de PDF con búsqueda semántica y Q&A."""

    # Parámetros por defecto (se pueden sobre‑escribir en el constructor)
    DEFAULT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_QA_MODEL = "deepset/roberta-base-squad2"
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_OVERLAP = 50

    # ──────────────────────────────────────────
    # Inicialización                                           
    # ──────────────────────────────────────────

    def __init__(
        self,
        embedder_name: str = DEFAULT_EMBEDDER,
        qa_model: str = DEFAULT_QA_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        print("🔁 Cargando modelos …")
        self.embedder = SentenceTransformer(embedder_name)
        self.qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_model)
        print("✅ Modelos cargados correctamente\n")

        # Configuración
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Contenedores de datos
        self.text_chunks: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.IndexFlatL2 | None = None
        self.full_text: str = ""
        self.tokens: List[str] = []  # Para estadísticas

    # ──────────────────────────────────────────
    # Carga y procesamiento del PDF                             
    # ──────────────────────────────────────────

    def _extract_text_from_pdf(self, pdf_file) -> str:
        """Extrae texto de cada página con PyPDF2 (ignora páginas sin texto)."""
        extracted = []
        reader = PyPDF2.PdfReader(pdf_file)
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text:
                extracted.append(f"\n--- Página {i} ---\n{page_text}")
        return "".join(extracted)

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Divide el texto en fragmentos solapados (palabras)."""
        words = text.split()
        step = self.chunk_size - self.overlap
        return [" ".join(words[i : i + self.chunk_size]) for i in range(0, len(words), step)]

    def _create_embeddings(self, chunks: List[str]):
        print("🤖 Generando embeddings y construyendo índice FAISS …")
        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        return embeddings, index

    def load_pdf(self, pdf_path: str | None = None) -> bool:
        """Carga un PDF y construye el índice vectorial.

        Si `pdf_path` es ``None`` y se está en Colab, se invoca
        `files.upload()` para permitir subir el archivo desde la interfaz.
        """
        # — 1. Obtener la ruta o subir archivo —
        if pdf_path is None:
            if files is None:
                print("❌ Debes proporcionar la ruta al PDF cuando no estás en Colab")
                return False
            print("📋 Sube tu archivo PDF:")
            uploaded = files.upload()
            if not uploaded:
                print("❌ No se subió ningún archivo")
                return False
            pdf_path = next(iter(uploaded))  # primera clave

        print(f"📖 Procesando PDF: {pdf_path}\n")

        # — 2. Extraer texto —
        try:
            with open(pdf_path, "rb") as f:
                self.full_text = self._extract_text_from_pdf(f)
        except Exception as exc:
            print(f"❌ Error leyendo el PDF: {exc}")
            return False

        if not self.full_text.strip():
            print("⚠ No se pudo extraer texto utilizable del PDF")
            return False

        # — 3. Limpieza, tokenización y división en chunks —
        cleaned = clean_text(self.full_text)
        self.tokens = tokenize(cleaned)
        self.text_chunks = self._split_text_into_chunks(cleaned)
        print(f"📄 Texto dividido en {len(self.text_chunks)} fragmentos\n")

        # — 4. Embeddings e índice —
        self.embeddings, self.index = self._create_embeddings(self.text_chunks)
        print("✅ PDF procesado correctamente\n")
        return True

    # ──────────────────────────────────────────
    # Búsqueda y Q&A                                            
    # ──────────────────────────────────────────

    def _find_relevant_chunks(self, question: str, top_k: int = 5):
        """Devuelve los índices y distancias de los *top‑k* chunks relevantes."""
        if self.index is None:
            return [], []
        query_emb = self.embedder.encode([question]).astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        return indices[0], distances[0]

    def answer_question(self, question: str, max_context: int = 1000, min_conf: float = 0.3) -> str:
        """Responde *question* usando los fragmentos más relevantes.

        ▸ *max_context* limita la longitud del contexto concatenado.  
        ▸ Devuelve una cadena con respuesta, confianza y extracto de contexto.
        """
        if not self.text_chunks:
            return "❌ Primero debes cargar un PDF"

        idxs, _ = self._find_relevant_chunks(question, top_k=5)
        context = " ".join(self.text_chunks[i] for i in idxs if i < len(self.text_chunks))
        context = context[:max_context]

        if not context.strip():
            return "⚠ No se encontró contexto relevante"

        result = self.qa_pipeline(question=question, context=context)
        answer, score = result["answer"], result["score"]

        response = (
            f"📝 **Respuesta:** {answer}\n\n"
            f"🎯 **Confianza:** {score:.2%}\n\n"
        )
        if score < min_conf:
            response += "⚠ *Confianza baja: interpreta con cautela*\n\n"
        response += "📚 **Contexto usado (truncado):**\n" + textwrap.indent(context[:300] + ("…" if len(context) > 300 else ""), "    ")
        return response

    # ──────────────────────────────────────────
    # Estadísticas rápidas                                      
    # ──────────────────────────────────────────

    def document_summary(self) -> str:
        if not self.full_text:
            return "❌ No hay documento cargado"
        wc, uc = len(self.tokens), len(set(self.tokens))
        return (
            "📊 **Resumen del documento**\n"
            f"• Palabras totales: {wc:,}\n"
            f"• Palabras únicas: {uc:,}\n"
            f"• Fragmentos: {len(self.text_chunks)}\n"
        )

    def most_common_tokens(self, n: int = 10):
        return Counter(self.tokens).most_common(n)

# ╝═════════════════════════════════════════════╝
# │  CLI / Ejecución directa                                   │
# ╚═════════════════════════════════════════════╚

def main():  # pragma: no cover
    print("🚀 Iniciando Analizador de PDF …")
    analyzer = PDFAnalyzer()

    # Cargar PDF (None → subida en Colab si aplica)
    if analyzer.load_pdf():
        print("\n" + "=" * 60)
        print(analyzer.document_summary())
        print("=" * 60 + "\n")
        print("💬 Ahora puedes hacer preguntas. Escribe 'salir' para terminar.")

        while True:
            question = input("🤔 Tu pregunta: ").strip()
            if question.lower() in {"salir", "exit", "quit"}:
                print("👋 ¡Hasta luego!")
                break
            if not question:
                print("⚠ Ingresa una pregunta válida")
                continue
            print("\n🔍 Buscando respuesta …\n")
            print(analyzer.answer_question(question))
            print()

    return analyzer

if __name__ == "__main__":
    main()
