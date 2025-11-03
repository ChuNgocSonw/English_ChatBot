import os
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv
import re
import wave
import io

# --- CẤU HÌNH ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

try:
    # Khởi tạo client để có thể import từ main.py
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    EMBEDDING_MODEL = "models/text-embedding-004"
except Exception as e:
    print(f"---!!! [LỖI] Không thể khởi tạo client trong retriever.py: {e} !!!---")
    raise e

def format_context(retrieved_data):
    """
    Định dạng dữ liệu tìm được và thêm thẻ span cho từ vựng.
    """
    context = ""
    
    if retrieved_data.get("vocabulary"):
        print("--- [LOG] Đang định dạng dữ liệu Vocabulary... ---")
        # Đổi nhãn thành tiếng Anh
        context += "Vocabulary Information:\n"
        for item in retrieved_data["vocabulary"]:
            word = item.get('word')
            context += f"- Word: <span class=\"tts-word\">{word}</span>\n"
            if item.get('phonetic'):
                context += f"  Phonetic: {item.get('phonetic')}\n"
            context += f"  Meaning: {item.get('meaning')}\n"
            if item.get('example'):
                context += f"  Example: {item.get('example')}\n"
    
    if retrieved_data.get("grammar"):
        print("--- [LOG] Đang định dạng dữ liệu Grammar... ---")
        # Đổi nhãn thành tiếng Anh
        context += "\nGrammar Information:\n"
        for item in retrieved_data["grammar"]:
            context += f"- Rule: {item.get('rule')}\n  Explanation: {item.get('explanation')}\n  Example: {item.get('example')}\n"
            
    if retrieved_data.get("idioms"):
        print("--- [LOG] Đang định dạng dữ liệu Idioms... ---")
        # Đổi nhãn thành tiếng Anh
        context += "\nIdiom Information:\n"
        for item in retrieved_data["idioms"]:
            context += f"- Phrase: {item.get('phrase')}\n  Meaning: {item.get('meaning')}\n  Example: {item.get('example')}\n"
    
    if retrieved_data.get("common_mistakes"):
        print("--- [LOG] Đang định dạng dữ liệu Common Mistakes... ---")
        # Đổi nhãn thành tiếng Anh
        context += "\nCommon Mistake Information:\n"
        for item in retrieved_data["common_mistakes"]:
            context += f"- Mistake: {item.get('mistake')}\n  Correction: {item.get('correction')}\n  Example: {item.get('example')}\n"

    if retrieved_data.get("conversations"):
        print("--- [LOG] Đang định dạng dữ liệu Conversations... ---")
        # Đổi nhãn thành tiếng Anh
        context += "\nConversation Example:\n"
        for item in retrieved_data["conversations"]:
            context += f"- Situation: {item.get('situation')}\n  Dialogue: {item.get('dialogue')}\n"
    
    if not context:
        print("--- [LOG] Không tìm thấy dữ liệu nào để định dạng. ---")
        
    return context.strip()

def search_context(search_term: str) -> str:
    """
    Hàm chính để tìm kiếm ngữ cảnh trong Supabase trên cả 5 bảng.
    """
    if not search_term:
        print("--- [LOG] Từ khóa tìm kiếm rỗng, bỏ qua truy vấn. ---")
        return ""
    try:
        print(f"--- [LOG] Đang tạo embedding cho từ khóa: '{search_term}' ---")
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=search_term,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        threshold = 0.65
        
        print(f"--- [LOG] Đang truy vấn 5 bảng với ngưỡng: {threshold} ---")
        
        # Gọi RPC cho cả 5 bảng
        vocab_res = supabase.rpc('match_vocabulary', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        grammar_res = supabase.rpc('match_grammar_rules', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        idioms_res = supabase.rpc('match_idioms', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        mistakes_res = supabase.rpc('match_common_mistakes', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        convo_res = supabase.rpc('match_conversations', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()

        print("--- [LOG] Đã hoàn tất truy vấn 5 bảng. ---")

        retrieved_data = {
            "vocabulary": vocab_res.data,
            "grammar": grammar_res.data,
            "idioms": idioms_res.data,
            "common_mistakes": mistakes_res.data,
            "conversations": convo_res.data
        }
        return format_context(retrieved_data)
    except Exception as e:
        print(f"---!!! [LỖI] Lỗi trong quá trình truy vấn (search_context): {e} !!!---")
        return ""