from fastapi import FastAPI
import os
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import List

app = FastAPI()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"
MODEL_NAME = "google/gemini-2.0-pro-exp-02-05:free"

SYSTEM_PROMPT = """أنت محامي مصري متخصص في جرائم السرقة وفقًا للمواد الواردة في القانون المصري.
الردود يجب أن:
1. تكون دقيقة قانونيًا مع الإشارة إلى المواد ذات الصلة.
2. تتبع الهيكل القانوني المناسب.
3. تحافظ على مستوى مهني واحترافي.
4. تستخدم المصطلحات القانونية الصحيحة.
5. تعتمد على النصوص القانونية من المستند المرفق (إن وجد).
6. يجب أن تقتصر الإجابة على الأسئلة القانونية فقط، ولا يجوز الإجابة على الأسئلة العامة التي لا تتعلق بالقانون."""

PDF_FILE_PATH = "D:/layer project/جريمة السرقة (1).pdf"

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                reshaped_text = arabic_reshaper.reshape(page_text)
                bidi_text = get_display(reshaped_text)
                text += bidi_text + "\n"
    return text

def split_text(text, chunk_size=4000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def find_relevant_chunk(question, chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [question])
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    best_chunk_index = similarities.argmax()
    return chunks[best_chunk_index]

@app.get("/load_pdf/")
def load_pdf():
    text = extract_text_from_pdf(PDF_FILE_PATH)
    chunks = split_text(text)
    return {"message": "تم استخراج النص بنجاح", "chunks": chunks}

@app.post("/ask_question/")
def ask_question(question: str):
    text = extract_text_from_pdf(PDF_FILE_PATH)
    chunks = split_text(text)
    context_text = SYSTEM_PROMPT
    if chunks:
        best_chunk = find_relevant_chunk(question, chunks)
        context_text += f"\n\nالمستند القانوني:\n{best_chunk}"

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": context_text}, {"role": "user", "content": question}], "temperature": 0.3, "max_tokens": 1000}
    
    response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
    response_data = response.json()
    
    if 'choices' in response_data and len(response_data['choices']) > 0:
        return {"response": response_data['choices'][0]['message']['content']}
    else:
        return {"error": "لم يتم استلام رد مناسب"}

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
