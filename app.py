import streamlit as st
import os
import requests
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure OpenRouter API
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.0-pro-exp-02-05:free"

uploaded_file = 'D:\layer project\جريمة السرقة (1).pdf'

# استخراج النص من PDF مع الحفاظ على اتصال الأحرف العربية

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1, layout=True)
            if page_text:
                reshaped_text = arabic_reshaper.reshape(page_text)
                bidi_text = get_display(reshaped_text)
                text += bidi_text + "\n"
    return text

# تقسيم النص إلى أجزاء صغيرة (Chunks)
def split_text(text, chunk_size=4000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# اختيار الجزء الأكثر ارتباطًا بالسؤال باستخدام TF-IDF
def find_relevant_chunk(question, chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks + [question])
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    best_chunk_index = similarities.argmax()
    return chunks[best_chunk_index], best_chunk_index

# إعداد النظام
SYSTEM_PROMPT = """أنت محامي مصري متخصص في جرائم السرقة وفقًا للمواد الواردة في القانون المصري.
الردود يجب أن:
1. تكون دقيقة قانونيًا مع الإشارة إلى المواد ذات الصلة.
2. تتبع الهيكل القانوني المناسب.
3. تحافظ على مستوى مهني واحترافي.
4. تستخدم المصطلحات القانونية الصحيحة.
5. تعتمد على النصوص القانونية من المستند المرفق (إن وجد).
6. يجب أن تقتصر الإجابة على الأسئلة القانونية فقط، ولا يجوز الإجابة على الأسئلة العامة التي لا تتعلق بالقانون."""

# معالجة الملف القانوني إذا تم تحميله
if uploaded_file:
    file_text = extract_text_from_pdf(uploaded_file)
    text_chunks = split_text(file_text)
    st.session_state["document_chunks"] = text_chunks

# تهيئة سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# دعم الاتجاه من اليمين لليسار (RTL)
st.markdown("""
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }
    p, div, h1, h2, h3, h4, h5, h6, span, label {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Arial', sans-serif !important;
    }
    .stChatInput textarea {
        direction: rtl !important;
        text-align: right !important;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("مستشارك القانوني")  
st.caption("نظام استشارات قانونية متخصص في القانون المصري ▪️")

# عرض الرسائل المخزنة
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# استقبال استفسار المستخدم
if prompt := st.chat_input("اطرح سؤالك القانوني هنا:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # اختيار الجزء الأكثر ارتباطًا بالسؤال
    context_text = SYSTEM_PROMPT
    if "document_chunks" in st.session_state:
        best_chunk, chunk_index = find_relevant_chunk(prompt, st.session_state["document_chunks"])
        context_text += f"\n\nالمستند القانوني (الجزء {chunk_index + 1} الأكثر صلة بالسؤال):\n{best_chunk}"

    # تجهيز الطلب للـ API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": context_text}] + st.session_state.messages[-5:],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    # استدعاء API مع مؤشر تحميل
    with st.spinner("جارٍ معالجة استفسارك..."):
        try:
            response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            if 'choices' in response_data and len(response_data['choices']) > 0:
                ai_response = response_data['choices'][0]['message']['content']
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            else:
                st.error("خطأ: لم يتم استلام رد مناسب من الخادم.")

        except requests.exceptions.HTTPError as http_err:
            st.error(f"خطأ HTTP: {http_err.response.status_code} - {http_err.response.text}")
        except KeyError as key_err:
            st.error(f"خطأ في معالجة البيانات: المفتاح {str(key_err)} غير موجود في الاستجابة")
        except Exception as e:
            st.error(f"خطأ غير متوقع: {str(e)}")
