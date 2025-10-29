import os
import json
import time
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# --- CẤU HÌNH ---
# Tải các biến môi trường từ file .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Kiểm tra biến môi trường
if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY]):
    raise ValueError("Lỗi: Vui lòng thiết lập các biến môi trường SUPABASE_URL, SUPABASE_SERVICE_KEY, và GOOGLE_API_KEY.")

# Khởi tạo client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Chọn mô hình embedding của Google
EMBEDDING_MODEL = "models/text-embedding-004"

def get_embedding(text):
    """Hàm gọi API của Google để tạo embedding cho một đoạn văn bản."""
    try:
        # Giảm thiểu khả năng lỗi "429: Resource has been exhausted"
        time.sleep(1) 
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT", # Quan trọng: Dùng cho dữ liệu sẽ được truy vấn sau này
            title="English Learning Data" # Tùy chọn nhưng nên có
        )
        return embedding['embedding']
    except Exception as e:
        print(f"Lỗi khi tạo embedding: {e}")
        return None

def generate_and_update_embeddings(table_name, text_columns):
    """Tạo và cập nhật embeddings cho một bảng cụ thể, có xử lý phân trang."""
    print(f"--- Bắt đầu xử lý bảng: {table_name} ---")

    offset = 0
    page_size = 1000  # Kích thước mỗi trang, khớp với giới hạn của Supabase
    total_updated = 0

    while True:
        print(f"\nĐang lấy dữ liệu từ hàng số {offset}...")
        
        # 1. Thêm .limit() và .offset() vào truy vấn
        response = (
            supabase.table(table_name)
            .select("id," + ",".join(text_columns))
            .is_("embedding", "null")
            .limit(page_size)
            .offset(offset)
            .execute()
        )
        
        data = response.data
        
        # 2. Điều kiện thoát vòng lặp: nếu không còn dữ liệu trả về
        if not data:
            print(f"✅ Không còn hàng nào cần cập nhật trong bảng {table_name}.")
            break

        print(f"Tìm thấy {len(data)} hàng cần cập nhật trong trang này.")

        for item in data:
            # Ghép các cột văn bản lại thành một chuỗi duy nhất
            combined_text = ". ".join([f"{col.capitalize()}: {item[col]}" for col in text_columns if item.get(col)])
            
            # Tạo embedding
            embedding_vector = get_embedding(combined_text)
            
            # Cập nhật lại vào Supabase nếu tạo embedding thành công
            if embedding_vector:
                try:
                    supabase.table(table_name).update({'embedding': embedding_vector}).eq('id', item['id']).execute()
                    print(f"  -> Đã cập nhật embedding cho ID: {item['id']}")
                    total_updated += 1
                except Exception as e:
                    print(f"  -> LỖI khi cập nhật embedding cho ID {item['id']}: {e}")

        # 3. Cập nhật offset cho trang tiếp theo
        offset += page_size

    print(f"\n🎉 Hoàn tất xử lý bảng: {table_name}. Tổng cộng đã cập nhật {total_updated} hàng.")


# --- Chạy script cho từng bảng ---
generate_and_update_embeddings('english_idioms', ['phrase', 'meaning', 'example'])
generate_and_update_embeddings('english_grammar_rules', ['rule', 'explanation', 'example'])
generate_and_update_embeddings('english_vocabulary', ['word', 'meaning', 'example'])