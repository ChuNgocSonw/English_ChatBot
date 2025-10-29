import http.server
import socketserver
import os

# --- CẤU HÌNH ---
PORT = 8080  # Cổng để chạy giao diện, ví dụ: http://localhost:8080
# THAY ĐỔI: Trỏ đến thư mục 'frontend' nơi chứa file index.html
DIRECTORY = "frontend" 

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# --- KHỞI CHẠY SERVER ---
# SỬA LỖI: Đưa 'Handler' vào làm đối số thứ hai của TCPServer
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.isdir(DIRECTORY):
        print(f"❌ LỖI: Không tìm thấy thư mục '{DIRECTORY}'. Vui lòng tạo thư mục và đặt file index.html vào trong đó.")
    else:
        print(f"✅ Giao diện đang được phục vụ từ thư mục '{DIRECTORY}' tại http://localhost:{PORT}")
        print("Mở trình duyệt và truy cập vào địa chỉ trên.")
        print("Nhấn Ctrl+C để dừng server.")
        httpd.serve_forever()

