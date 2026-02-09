import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

PDF_PATH = "data/document.pdf"
VECTOR_DIR = "vectors"

# 1. Load PDF
reader = PdfReader(PDF_PATH)
documents = []

for page_no, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
        documents.append(
            Document(
                page_content=text,
                metadata={"page": page_no}
            )
        )

print(f"Loaded {len(documents)} pages")

# 2. Load embedding model (LangChain wrapper)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 3. Build FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Save vectors
os.makedirs(VECTOR_DIR, exist_ok=True)
vectorstore.save_local(VECTOR_DIR)

print("FAISS vectors saved to ./vectors")
