import io
import traceback
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import pytesseract
import pandas as pd
from rag_system import RAGSystem

app = FastAPI()
templates = Jinja2Templates(directory="templates")
rag = RAGSystem()

def extract_text(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='ara+eng').strip()
        print(f"📝 OCR Text: {text[:100]}")
        return text
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_csv/")
async def upload_csv(
    price_list: UploadFile = File(None),
    policy_rules: UploadFile = File(None),
    coverage_rules: UploadFile = File(None),
    patient_db: UploadFile = File(None),
    provider_registry: UploadFile = File(None),
    claims_history: UploadFile = File(None),
    clinical_guidelines: UploadFile = File(None),
    watchlist: UploadFile = File(None)
):
    try:
        files = {}
        for name, file in [
            ('price_list', price_list),
            ('policy_rules', policy_rules),
            ('coverage_rules', coverage_rules),
            ('patient_db', patient_db),
            ('provider_registry', provider_registry),
            ('claims_history', claims_history),
            ('clinical_guidelines', clinical_guidelines),
            ('watchlist', watchlist)
        ]:
            if file:
                try:
                    df = pd.read_csv(io.BytesIO(await file.read()))
                    files[name] = df
                    print(f"✅ Loaded {name}: {len(df)} rows")
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
        
        if files:
            success = rag.load_csv_to_rag(files)
            if success:
                return {"status": "success", "files_loaded": list(files.keys())}
        
        return {"status": "error", "files_loaded": []}
        
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/validate/")
async def validate(invoice: UploadFile = File(...)):
    try:
        print(f"📤 Received invoice: {invoice.filename}")
        
        # Read image
        image_bytes = await invoice.read()
        print(f"📦 Image size: {len(image_bytes)} bytes")
        
        # Extract text
        ocr_text = extract_text(image_bytes)
        print(f"📝 OCR length: {len(ocr_text)}")
        
        if not ocr_text:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "فشل استخراج النص من الصورة",
                    "rag_analysis": {
                        "status": "error",
                        "claimed_amount": 0,
                        "approved_amount": 0,
                        "confidence": 0,
                        "checks": {},
                        "warnings": ["لم يتم العثور على نص في الصورة"]
                    }
                }
            )
        
        # Analyze
        analysis = rag.analyze_claim(ocr_text)
        print(f"✅ Analysis: {analysis['status']}")
        
        return {
            "invoice_filename": invoice.filename,
            "ocr_text": ocr_text[:300],
            "rag_analysis": analysis
        }
        
    except Exception as e:
        print(f"❌ Validate Error: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "rag_analysis": {
                    "status": "error",
                    "claimed_amount": 0,
                    "approved_amount": 0,
                    "confidence": 0,
                    "checks": {},
                    "warnings": [f"خطأ: {str(e)}"]
                }
            }
        )

@app.get("/health/")
def health():
    return {
        "status": "healthy",
        "rag_loaded": rag.vectorstore is not None
    }
