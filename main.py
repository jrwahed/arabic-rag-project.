# main.py - نسخة مبسطة بدون مشاكل
import os
import io
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PIL import Image
import pytesseract

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="Arabic RAG API")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

try:
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info("✅ Components ready")
except Exception as e:
    logging.error(f"❌ Error: {e}")
    raise

vectorstore = None

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image, lang='ara+eng', config='--psm 6')
        return text.strip()
    except Exception as e:
        raise Exception(f"OCR failed: {e}")

@app.get("/")
def root():
    return {"message": "🚀 Arabic RAG API", "status": "ready"}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "يجب رفع صورة")
    
    try:
        file_content = await file.read()
        text = extract_text_from_image(file_content)
        
        if not text:
            raise HTTPException(400, "لم يتم العثور على نص")
        
        docs = text_splitter.create_documents([text], metadatas=[{"source": file.filename}])
        vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./db")
        
        return {
            "status": "success",
            "message": f"✅ تم رفع {file.filename}",
            "text_preview": text[:300]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not vectorstore:
        raise HTTPException(400, "يجب رفع صورة أولاً")
    
    try:
        docs = vectorstore.similarity_search(question, k=3)
        
        if not docs:
            return {"answer": "لم أجد معلومات"}
        
        # إرجاع النص مباشرة بدون LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "question": question,
            "answer": f"المعلومات المتعلقة بسؤالك:\n\n{context[:1000]}",
            "sources": [doc.metadata.get("source") for doc in docs]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health/")
def health():
    return {"status": "healthy", "database": "ready" if vectorstore else "empty"}
