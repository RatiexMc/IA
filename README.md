# IA

Este repositorio contiene un ejemplo sencillo para analizar documentos PDF
usando Python. El archivo `pdf_analyzer.py` reúne y simplifica las
funcionalidades de los códigos `mio.py` y del proyecto original del profesor:

- Conversión de PDF a texto y limpieza básica.
- Tokenización y conteo de frecuencia de palabras.
- Creación de embeddings con `sentence-transformers` y un índice FAISS.
- Sistema de preguntas y respuestas con el modelo gratuito
  `deepset/roberta-base-squad2`.

## Uso rápido en Google Colab

1. Instala las dependencias:

   ```python
   !pip install PyPDF2 sentence-transformers faiss-cpu transformers torch
   ```

2. Carga y procesa tu PDF:

   ```python
   from pdf_analyzer import PDFAnalyzer

   analyzer = PDFAnalyzer()
   analyzer.load_pdf('mi_documento.pdf')
   print(analyzer.document_summary())
   print(analyzer.most_common_tokens(10))
   ```

3. Realiza consultas sobre el texto:

   ```python
   respuesta = analyzer.answer_question('¿Cuál es el tema principal?')
   print(respuesta)
   ```

Todo el flujo se puede ejecutar de forma gratuita dentro de Colab,
sin necesidad de pagar por modelos externos.
