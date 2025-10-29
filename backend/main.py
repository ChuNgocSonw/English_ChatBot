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
    prompt = f"""
    Detect the language of the following text. Respond with either 'Vietnamese' or 'English'. 
    If you are unsure, default to 'Vietnamese'.

    Text: "Lament có nghĩa là gì?" -> Language: Vietnamese
    Text: "what is the meaning of frugal?" -> Language: English
    Text: "{user_query}" -> Language:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        language = response.text.strip()
        print(f"Ngôn ngữ được phát hiện: '{language}'")
        return "English" if "english" in language.lower() else "Vietnamese"
    except Exception as e:
        print(f"Lỗi khi phát hiện ngôn ngữ: {e}")
        return "Vietnamese"

def determine_intent(user_query: str) -> str:
    """
    Sử dụng LLM để phân loại ý định của người dùng.
    """
    prompt = f"""
    Phân loại câu hỏi của người dùng thành "Q&A" (hỏi kiến thức) hoặc "Conversational" (trò chuyện thông thường).

    Ví dụ:
    - "Flagrant nghĩa là gì?" -> Q&A
    - "chào bạn" -> Conversational
    - "cảm ơn" -> Conversational
    - "cho tôi ví dụ về 'a piece of cake'" -> Q&A

    Câu hỏi: "{user_query}"
    Loại:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        intent = response.text.strip()
        print(f"Đã xác định ý định: '{intent}'")
        return intent if intent in ["Q&A", "Conversational"] else "Q&A"
    except Exception as e:
        print(f"Lỗi khi xác định ý định: {e}")
        return "Q&A"

def extract_keyword(user_query: str) -> str:
    """
    Trích xuất từ khóa tiếng Anh từ một câu hỏi Q&A.
    """
    prompt = f"""
    Extract the main English keyword or phrase from the following query. Return only the keyword. If there is no English keyword, return an empty string.

    Examples:
    - "Flagrant nghĩa là gì và cho câu ví dụ" -> Flagrant
    - "cho tôi ví dụ về 'a piece of cake'" -> a piece of cake
    - "xin chào bạn" -> 

    Query: "{user_query}"
    Keyword:
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        keyword = response.text.strip().replace('"', '')
        print(f"Đã trích xuất từ khóa: '{keyword}'")
        return keyword
    except Exception as e:
        print(f"Lỗi khi trích xuất từ khóa: {e}")
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
        # BƯỚC 1: Xác định ý định và ngôn ngữ
        intent = determine_intent(query.text)
        detected_language = detect_language(query.text)

        # KỊCH BẢN 1: Người dùng đang trò chuyện (ƯU TIÊN HÀNG ĐẦU)
        if intent == "Conversational":
            prompt = f"You are a friendly English tutor chatbot named English AI Tutor. Respond conversationally to the user's message in {detected_language}. Keep it natural and brief. User message: '{query.text}'"
            response = GENERATION_MODEL.generate_content(prompt)
            return {"answer": response.text, "source_context": "Conversational"}

        # KỊCH BẢN 2: Người dùng hỏi kiến thức (Q&A)
        # Chỉ trích xuất từ khóa nếu là Q&A
        search_term = extract_keyword(query.text)
        
        # Nếu không trích xuất được từ khóa từ câu hỏi Q&A => Xử lý như trò chuyện
        if not search_term:
            prompt = f"You are a friendly English tutor. The user asked a question, possibly Q&A, but no clear English keyword was found: '{query.text}'. Respond helpfully in {detected_language}, guiding them to ask about a specific English word, grammar rule, or idiom."
            response = GENERATION_MODEL.generate_content(prompt)
            return {"answer": response.text, "source_context": "Conversational Fallback"}

        # Nếu có từ khóa, tiến hành tìm kiếm
        context_string = search_context(search_term)
        
        prompt_template = ""
        if context_string:
            # Tìm thấy thông tin
            prompt_template = f"""
            You are an expert English tutor. Use the provided context to directly answer the user's question.
            
            CRITICAL RULES:
            1. Preserve HTML tags like `<span class="tts-word">...</span>` EXACTLY as they appear in the context. DO NOT add new ones.
            2. Include phonetic transcription if available.
            3. Provide bilingual format for meanings/examples (English first, then Vietnamese in parentheses).
            4. Respond in {detected_language}. Be concise.

            Context: {context_string}
            User's Question: {query.text}
            Answer (in {detected_language}):
            """
        else:
            # Không tìm thấy thông tin
            prompt_template = f"""
            You are a friendly English tutor. Inform the user you couldn't find information for "{search_term}". 
            Suggest they try another keyword or check spelling. Respond in {detected_language}.
            """
        
        response = GENERATION_MODEL.generate_content(prompt_template)
        return {"answer": response.text, "source_context": context_string if context_string else "Fallback"}

    except Exception as e:
        print(f"Lỗi máy chủ nội bộ: {e}")
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
            print(f"CACHE HIT: Tìm thấy file '{file_path}'.")
            return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}

        print(f"CACHE MISS: Tạo file '{file_path}'.")
        
        max_retries = 2
        retry_delay = 40

        for attempt in range(max_retries):
            try:
                tts_prompt = f"Speak the following word clearly: {request.text}"
                tts_config = {"response_modalities": ["AUDIO"], "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}}
                
                response = TTS_MODEL.generate_content(tts_prompt, generation_config=tts_config)
                
                if response.candidates and response.candidates[0].content.parts:
                    audio_part = response.candidates[0].content.parts[0]
                    pcm_data = base64.b64decode(audio_part.inline_data.data) # Decode base64 here
                    mime_type = audio_part.inline_data.mime_type
                    
                    sample_rate_match = re.search(r'rate=(\d+)', mime_type)
                    if not sample_rate_match: raise ValueError("Sample rate not found.")
                    sample_rate = int(sample_rate_match.group(1))

                    wav_data = pcm_to_wav_bytes(pcm_data, sample_rate)

                    supabase.storage.from_(AUDIO_BUCKET).upload(file=wav_data, path=file_path, file_options={"content-type": "audio/wav", "x-upsert": "true"})
                    print(f"Đã tải lên file '{file_path}'.")

                    return {"audioUrl": supabase.storage.from_(AUDIO_BUCKET).get_public_url(file_path)}
                else:
                     # Thêm kiểm tra này để xử lý trường hợp không có candidates hợp lệ ngay cả khi không có lỗi rõ ràng
                    print("Lỗi TTS: API không trả về candidate hợp lệ (lý do không xác định).")
                    if response.prompt_feedback:
                        print(f"Lý do phản hồi API: {response.prompt_feedback}")
                    # Nếu đã thử hết số lần mà vẫn lỗi, ném ra lỗi cuối cùng
                    if attempt == max_retries - 1:
                        raise HTTPException(status_code=500, detail="TTS generation failed after multiple retries.")
                    # Nếu chưa hết số lần thử, đợi và thử lại (chủ yếu cho lỗi quota)
                    else:
                         print(f"Đang đợi {retry_delay} giây để thử lại...")
                         time.sleep(retry_delay)
                         continue # Thử lại vòng lặp
                
            except Exception as e:
                # Kiểm tra xem có phải lỗi quota không trước khi thử lại
                is_quota_error = "429" in str(e) or "quota" in str(e).lower()
                if is_quota_error and attempt < max_retries - 1:
                    print(f"Lỗi Quota. Đang đợi {retry_delay} giây để thử lại...")
                    time.sleep(retry_delay)
                    continue # Thử lại vòng lặp
                else:
                    # Nếu là lỗi khác hoặc hết lần thử, ném lỗi ra ngoài
                    raise e # Ném lỗi gốc để có thông tin chi tiết
        
        # Nếu hết vòng lặp mà vẫn lỗi (thường là do quota)
        raise HTTPException(status_code=429, detail="API quota exceeded after retries.")

    except Exception as e:
        print(f"Lỗi khi tạo hoặc lấy âm thanh: {e}")
        # Trả về lỗi cụ thể hơn nếu có thể
        error_detail = str(e) if isinstance(e, ValueError) else "Failed to process speech."
        raise HTTPException(status_code=500, detail=error_detail)