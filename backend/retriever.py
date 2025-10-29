import os
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# --- CẤU HÌNH ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

try:
    # === THAY ĐỔI: Khởi tạo client để có thể import từ main.py ===
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    EMBEDDING_MODEL = "models/text-embedding-004"
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo các client. Lỗi: {e}")
    raise e

def format_context(retrieved_data):
    """
    Định dạng dữ liệu tìm được và thêm thẻ span cho từ vựng.
    """
    context = ""
    if retrieved_data.get("vocabulary"):
        context += "\nThông tin về từ vựng:\n"
        for item in retrieved_data["vocabulary"]:
            word = item.get('word')
            context += f"- Word: <span class=\"tts-word\">{word}</span>\n"
            # Thêm các thông tin khác
            if item.get('phonetic'):
                context += f"  Phonetic: {item.get('phonetic')}\n"
            context += f"  Meaning: {item.get('meaning')}\n  Example: {item.get('example')}\n"
    
    # Thêm logic cho grammar và idioms nếu cần
    if retrieved_data.get("grammar"):
        context += "\nThông tin về ngữ pháp:\n"
        for item in retrieved_data["grammar"]:
            context += f"- Rule: {item.get('rule')}\n  Explanation: {item.get('explanation')}\n  Example: {item.get('example')}\n"
            
    if retrieved_data.get("idioms"):
        context += "\nThông tin về thành ngữ:\n"
        for item in retrieved_data["idioms"]:
            context += f"- Phrase: {item.get('phrase')}\n  Meaning: {item.get('meaning')}\n  Example: {item.get('example')}\n"
    
    return context.strip()

def search_context(search_term: str) -> str:
    if not search_term:
        return ""
    try:
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=search_term,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        threshold = 0.65
        
        # Tìm kiếm ở cả 3 bảng
        vocab_res = supabase.rpc('match_vocabulary', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        grammar_res = supabase.rpc('match_grammar_rules', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()
        idioms_res = supabase.rpc('match_idioms', {'query_embedding': query_embedding, 'match_threshold': threshold, 'match_count': 1}).execute()

        retrieved_data = { 
            "vocabulary": vocab_res.data,
            "grammar": grammar_res.data,
            "idioms": idioms_res.data
        }
        return format_context(retrieved_data)
    except Exception as e:
        print(f"Lỗi trong quá trình truy vấn: {e}")
        return ""