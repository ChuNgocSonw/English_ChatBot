import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import base64
import re
import wave
import io
import time

# Sử dụng relative import để đảm bảo hoạt động chính xác
from .retriever import search_context, supabase

# --- CẤU HÌNH ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

GENERATION_MODEL = genai.GenerativeModel('gemini-2.5-flash')
TTS_MODEL = genai.GenerativeModel('gemini-2.5-flash-preview-tts')


app = FastAPI()

# --- Thêm CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Các Model Dữ liệu ---
class Query(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

# --- CÁC HÀM XỬ LÝ LOGIC ---
def extract_keyword(user_query: str) -> str:
    prompt = f"""
    Extract the main English keyword or phrase from the following query. Return only the keyword.
    Query: "{user_query}"
    Keyword:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        return response.text.strip().replace('"', '')
    except Exception:
        return ""

def pcm_to_wav_bytes(pcm_data, sample_rate):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return wav_buffer.getvalue()

# --- API ENDPOINTS ---
@app.post("/answer")
def get_answer(query: Query):
    try:
        search_term = extract_keyword(query.text)
        if not search_term:
            prompt = f"You are a friendly English tutor. Respond to '{query.text}' in Vietnamese."
            response = GENERATION_MODEL.generate_content(prompt)
            return {"answer": response.text, "source_context": "Conversational"}

        context_string = search_context(search_term)
        if context_string:
            # === THAY ĐỔI QUAN TRỌNG Ở ĐÂY ===
            prompt_template = f"""
            You are an expert English tutor. Use the provided context to answer the user's question in Vietnamese.

            **CRITICAL RULES:**
            1.  **Preserve HTML:** You MUST preserve HTML tags like `<span class="tts-word">...</span>` **ONLY IF** they appear within the 'Thông tin về từ vựng:' section of the Context.
            2.  **DO NOT ADD NEW `<span class="tts-word">...</span>` tags ANYWHERE.** Even if you see an English word, do not wrap it unless it was already wrapped in the context's vocabulary section.
            3.  **Bilingual Format:** Provide English first, then Vietnamese translation in parentheses for meanings/examples.
            4.  **Structure:** Start with a simple Vietnamese intro sentence WITHOUT any HTML tags or Markdown. Then, present the details from the context using Markdown list format (`- **Label:** ...`).
            5.  Include Phonetic if available in the context.
            6.  Be concise and accurate.

            **Context:**
            {context_string}
            
            **User's question:**
            {query.text}

            **Your answer (following all rules strictly):**
            """
        else:
            prompt_template = f"You are a friendly English tutor. Inform the user you couldn't find info for '{search_term}'. Respond in Vietnamese."
        
        response = GENERATION_MODEL.generate_content(prompt_template)
        # Thêm một bước làm sạch cuối cùng để đảm bảo (phòng ngừa)
        answer_text = response.text
        # Chỉ giữ lại thẻ span nếu nó nằm ngay sau "- Word:"
        def keep_word_span(match):
            return f"- Word: <span class=\"tts-word\">{match.group(1)}</span>"
        # Xóa tất cả các thẻ span khác trước
        answer_text_cleaned = re.sub(r'<span class="tts-word">(.*?)</span>', r'\1', answer_text)
        # Khôi phục lại thẻ span đúng vị trí
        answer_text_final = re.sub(r'-\s*\*\*Word:\*\*\s*(.*?)\n', keep_word_span, answer_text_cleaned, flags=re.IGNORECASE)

        return {"answer": answer_text_final, "source_context": context_string if context_string else "Fallback"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize-speech")
def synthesize_speech(request: TTSRequest):
    """
    Kiểm tra cache, tạo file nghe nếu cần, lưu vào Storage và trả về URL.
    """
    AUDIO_BUCKET = 'audio_cache'
    try:
        sanitized_text = re.sub(r'[^a-z0-9]', '_', request.text.lower())
        file_path = f"{sanitized_text}.wav"

        file_list = supabase.storage.from_(AUDIO_BUCKET).list(path="", options={"search": file_path})

        if file_list:
            print(f"CACHE HIT: Found '{file_path}'.")
            return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}

        print(f"CACHE MISS: Creating '{file_path}'.")
        
        max_retries = 2
        retry_delay = 40

        for attempt in range(max_retries):
            try:
                tts_prompt = f"Speak the following word clearly: {request.text}"
                tts_config = {"response_modalities": ["AUDIO"], "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}}
                
                response = TTS_MODEL.generate_content(tts_prompt, generation_config=tts_config)
                
                if response.candidates and response.candidates[0].content.parts:
                    audio_part = response.candidates[0].content.parts[0]
                    pcm_data = audio_part.inline_data.data
                    mime_type = audio_part.inline_data.mime_type
                    
                    sample_rate_match = re.search(r'rate=(\d+)', mime_type)
                    if not sample_rate_match: raise ValueError("Sample rate not found.")
                    sample_rate = int(sample_rate_match.group(1))

                    wav_data = pcm_to_wav_bytes(pcm_data, sample_rate)

                    supabase.storage.from_(AUDIO_BUCKET).upload(file=wav_data, path=file_path, file_options={"content-type": "audio/wav", "x-upsert": "true"})
                    print(f"Uploaded '{file_path}'.")

                    return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}
                
            except Exception as e:
                # Kiểm tra xem có phải lỗi quota không trước khi thử lại
                is_quota_error = "429" in str(e) or "quota" in str(e).lower()
                if is_quota_error and attempt < max_retries - 1:
                    print(f"Lỗi Quota. Đang đợi {retry_delay} giây để thử lại...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Nếu là lỗi khác hoặc hết lần thử, ném lỗi ra ngoài
                    raise e
        
        # Nếu hết vòng lặp mà vẫn lỗi (do quota)
        raise HTTPException(status_code=429, detail="API quota exceeded after retries.")

    except Exception as e:
        print(f"Lỗi khi tạo hoặc lấy âm thanh: {e}")
        # Trả về lỗi cụ thể hơn nếu có thể
        error_detail = str(e) if isinstance(e, ValueError) else "Failed to process speech."
        raise HTTPException(status_code=500, detail=error_detail)