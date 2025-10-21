
# 🚀 Arabic RAG API

نظام ذكاء اصطناعي لاستخراج النصوص من الصور والإجابة على الأسئلة بالعربية

## ⭐ المميزات

- 📸 استخراج النص من الصور (OCR)
- 🇦🇪 دعم اللغة العربية والإنجليزية
- 🔍 البحث الدلالي في النصوص
- 💾 قاعدة بيانات محلية (ChromaDB)
- ⚡ سريع وسهل الاستخدام

## 🛠️ التثبيت

ثبت المكتبات
pip install -r requirements.txt

ثبت Tesseract OCR
sudo apt-get install tesseract-ocr tesseract-ocr-ara



## 🔑 الإعدادات

أنشئ ملف `.env`:

HUGGINGFACE_API_KEY=your_key_here



## ▶️ التشغيل

uvicorn main:app --host 0.0.0.0 --port 8000



افتح: `http://localhost:8000/docs`

## 📝 الاستخدام

1. **رفع صورة:** `POST /upload/`
2. **طرح سؤال:** `POST /ask/`
3. **فحص الصحة:** `GET /health/`

## 👨‍💻 المطور

**محمد وحيد** - متخصص في تطوير الأعمال والتسويق الرقمي

تم التطوير في أكتوبر 2025

## 📄 الترخيص

MIT License - استخدم المشروع كما تشاء 😊
احفظ:

bash
Ctrl + O
Enter
Ctrl + X
ارفعه على GitHub:

bash
git add README.md
git commit -m "إضافة ملف README"
git push
