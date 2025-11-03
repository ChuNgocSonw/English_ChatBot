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

def detect_language(user_query: str) -> str:
    """Sử dụng Gemini để phát hiện ngôn ngữ của câu hỏi."""
    print(f"--- [LOG] Bắt đầu phát hiện ngôn ngữ cho: '{user_query}' ---")
    prompt = f"""
    Detect the language of the following text. Respond with ONLY 'Vietnamese' or 'English'. 
    If unsure, default to 'Vietnamese'.
    Text: "{user_query}"
    Language:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        language = response.text.strip().replace("'", "").replace('"', '')
        
        if "english" in language.lower():
            print(f"--- [LOG] Ngôn ngữ được phát hiện: 'English' ---")
            return "English"
        
        print(f"--- [LOG] Ngôn ngữ được phát hiện: 'Vietnamese' ---")
        return "Vietnamese"
    except Exception as e:
        print(f"--- [LỖI] Khi phát hiện ngôn ngữ: {e} ---")
        return "Vietnamese"

def determine_intent(user_query: str) -> str:
    """
    Sử dụng LLM để phân loại ý định của người dùng một cách đáng tin cậy.
    """
    print(f"--- [LOG] Bắt đầu phân loại ý định cho: '{user_query}' ---")
    # Thêm các ví dụ dễ nhầm lẫn để huấn luyện AI
    prompt = f"""
    Classify the user's query into "Q&A" (asking for knowledge) or "Conversational" (small talk).

    Examples:
    - "What does ubiquitous mean?" -> Q&A
    - "cho tôi ví dụ về 'break a leg'" -> Q&A
    - "xin chào" -> Conversational
    - "hello" -> Conversational
    - "hi" -> Conversational
    - "cảm ơn bạn" -> Conversational
    - "bạn là ai?" -> Conversational
    - "tôi muốn học tiếng anh" -> Conversational
    - "thì hiện tại đơn" -> Q&A
    - "Frugal có nghĩa là gì vậy" -> Q&A
    - "hội thoại đặt đồ ăn" -> Q&A

    Query: "{user_query}"
    Classification:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        intent = response.text.strip()
        print(f"--- [LOG] Đã xác định ý định: '{intent}' ---")
        return intent if intent in ["Q&A", "Conversational"] else "Q&A"
    except Exception as e:
        print(f"--- [LỖI] Khi xác định ý định: {e} ---")
        return "Q&A"

def extract_keyword(user_query: str) -> str:
    """Trích xuất từ khóa tiếng Anh từ một câu hỏi Q&A."""
    print(f"--- [LOG] Bắt đầu trích xuất từ khóa từ: '{user_query}' ---")
    prompt = f"""
    Extract the main English keyword or phrase from the following query. Return only the keyword.
    Query: "{user_query}"
    Keyword:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        keyword = response.text.strip().replace('"', '')
        print(f"--- [LOG] Đã trích xuất từ khóa: '{keyword}' ---")
        return keyword
    except Exception as e:
        print(f"--- [LỖI] Khi trích xuất từ khóa: {e} ---")
        return ""

def pcm_to_wav_bytes(pcm_data, sample_rate):
    """Chuyển đổi dữ liệu PCM thô thành định dạng WAV trong bộ nhớ."""
    print("--- [LOG] Đang chuyển đổi PCM sang WAV... ---")
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    print("--- [LOG] Đã chuyển đổi PCM sang WAV thành công. ---")
    return wav_buffer.getvalue()

# --- API ENDPOINTS ---
@app.post("/answer")
def get_answer(query: Query):
    print(f"\n--- [LOG] Nhận được yêu cầu /answer: '{query.text}' ---")
    try:
        # === THAY ĐỔI LOGIC: Bước 1 là phân loại ý định ===
        intent = determine_intent(query.text)
        detected_language = detect_language(query.text)
        
        # === KỊCH BẢN 1: Người dùng đang trò chuyện (ƯU TIÊN HÀNG ĐẦU) ===
        if intent == "Conversational":
            print("--- [LOG] Xử lý yêu cầu dạng: Conversational. ---")
            prompt = f"You are a friendly English tutor chatbot named English AI Tutor. Respond conversationally to the user's message in {detected_language}. Keep it natural and brief. User message: '{query.text}'"
            response = GENERATION_MODEL.generate_content(prompt)
            print("--- [LOG] Đã tạo phản hồi 'Conversational'. ---")
            return {"answer": response.text, "source_context": "Conversational"}

        # === KỊCH BẢN 2: Người dùng đang hỏi kiến thức (Q&A) ===
        print("--- [LOG] Xử lý yêu cầu dạng: Q&A. ---")
        search_term = extract_keyword(query.text)
        
        if not search_term:
            # Nếu là Q&A nhưng không có từ khóa (ví dụ: câu hỏi quá chung chung)
            print("--- [LOG] Ý định Q&A nhưng không tìm thấy từ khóa. Chuyển sang Fallback. ---")
            prompt = f"You are a friendly English tutor. The user asked: '{query.text}'. Respond helpfully in {detected_language}, guiding them to ask about a specific English word, grammar rule, or idiom. Answer in {detected_language}."
            response = GENERATION_MODEL.generate_content(prompt)
            return {"answer": response.text, "source_context": "Conversational Fallback"}

        # Nếu có từ khóa, tiến hành tìm kiếm
        print(f"--- [LOG] Đang tìm kiếm ngữ cảnh cho: '{search_term}' ---")
        context_string = search_context(search_term)
        
        if context_string:
            print("--- [LOG] Đã tìm thấy ngữ cảnh. Đang tạo phản hồi RAG. ---")
            prompt_template = f"""
            You are an expert English tutor. Your task is to provide a comprehensive, bilingual answer based on the context, following a strict format.

            **CRITICAL RULES:**
            1.  Preserve HTML tags (e.g., `<span class="tts-word">...</span>`) EXACTLY as they appear in the context.
            2.  DO NOT add new `tts-word` tags.
            3.  Provide bilingual format (English and Vietnamese) for meanings/examples.
            4.  Include phonetics if available.
            5.  Start with a simple intro sentence.
            6.  Respond in {detected_language}.

            **REQUIRED RESPONSE STRUCTURE EXAMPLE:**
            Chào bạn! Từ "Superfluous" có ý nghĩa như sau:

            - **Word:** <span class="tts-word">Superfluous</span>
            - **Phonetic:** /suːˈpɜː.flu.əs/
            - **Meaning:** Unnecessary, especially through being more than enough. (Không cần thiết, đặc biệt là khi nó nhiều hơn mức đủ.)
            - **Example:** The report contained superfluous information that confused readers. (Bản báo cáo chứa thông tin thừa thãi làm độc giả bối rối.)
            ---
            **Context:**
            {context_string}
            ---
            **User's question:**
            {query.text}
            ---
            **Your answer (in {detected_language}, following all rules and structure):**
            """
        else:
            print(f"--- [LOG] Không tìm thấy ngữ cảnh cho '{search_term}'. Đang tạo phản hồi Fallback. ---")
            prompt_template = f"You are a friendly English tutor. Inform the user you couldn't find info for '{search_term}'. Respond in {detected_language}."
        
        response = GENERATION_MODEL.generate_content(prompt_template)
        print("--- [LOG] Đã tạo phản hồi từ AI. ---")
        return {"answer": response.text, "source_context": context_string if context_string else "Fallback"}
    except Exception as e:
        print(f"---!!! [LỖI] Lỗi máy chủ nội bộ trong /answer: {e} !!!---")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize-speech")
def synthesize_speech(request: TTSRequest):
    """
    Kiểm tra cache, tạo file nghe nếu cần, lưu vào Storage và trả về URL.
    """
    AUDIO_BUCKET = 'audio_cache'
    print(f"\n--- [LOG] Nhận được yêu cầu /synthesize-speech cho: '{request.text}' ---")
    try:
        sanitized_text = re.sub(r'[^a-z0-9]', '_', request.text.lower())
        file_path = f"{sanitized_text}.wav"

        print(f"--- [LOG] Đang kiểm tra cache cho file: '{file_path}' ... ---")
        file_list = supabase.storage.from_(AUDIO_BUCKET).list(path="", options={"search": file_path})

        if file_list:
            print(f"--- [LOG] CACHE HIT: Tìm thấy file '{file_path}'. Trả về URL. ---")
            return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}

        print(f"--- [LOG] CACHE MISS: Không tìm thấy file. Đang tạo mới... ---")
        
        max_retries = 2
        retry_delay = 40

        for attempt in range(max_retries):
            try:
                tts_prompt = f"Speak the following word clearly: {request.text}"
                tts_config = {"response_modalities": ["AUDIO"], "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}}
                
                print(f"--- [LOG] Đang gọi API TTS của Google (Lần {attempt + 1})... ---")
                response = TTS_MODEL.generate_content(tts_prompt, generation_config=tts_config)
                
                if response.candidates and response.candidates[0].content.parts:
                    print("--- [LOG] Đã nhận được dữ liệu âm thanh từ API. ---")
                    audio_part = response.candidates[0].content.parts[0]
                    
                    # === SỬA LỖI QUAN TRỌNG: Giải mã (decode) Base64 ===
                    # 1. Dữ liệu âm thanh là chuỗi Base64 (str), CẦN GIẢI MÃ (decode) về bytes thô
                    pcm_data_bytes = base64.b64decode(audio_part.inline_data.data) 
                    
                    # 2. Dữ liệu mime type đã là (str), KHÔNG CẦN GIẢI MÃ
                    mime_type_string = audio_part.inline_data.mime_type
                    
                    # 3. Trích xuất sample rate từ string
                    sample_rate_match = re.search(r'rate=(\d+)', mime_type_string)
                    if not sample_rate_match: raise ValueError("Sample rate not found.")
                    sample_rate = int(sample_rate_match.group(1))

                    # 4. Chuyển đổi âm thanh thô (bytes) sang WAV (bytes)
                    wav_data = pcm_to_wav_bytes(pcm_data_bytes, sample_rate)

                    print(f"--- [LOG] Đang tải file '{file_path}' lên Supabase Storage... ---")
                    supabase.storage.from_(AUDIO_BUCKET).upload(
                        file=wav_data,
                        path=file_path,
                        file_options={"content-type": "audio/wav", "x-upsert": "true"}
                    )
                    print(f"--- [LOG] Đã tải lên file thành công. ---")

                    # 5. Trả về URL cho client
                    return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"--- [LỖI] Lỗi Quota 429. Đang đợi {retry_delay} giây để thử lại... ---")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
        
        raise HTTPException(status_code=429, detail="API quota exceeded.")

    except Exception as e:
        print(f"---!!! [LỖI] Lỗi khi tạo hoặc lấy âm thanh: {e} !!!---")
        raise HTTPException(status_code=500, detail="Failed to process speech.")