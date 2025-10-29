import csv
import json
import os # Thư viện để làm việc với tên file

def convert_csv_to_json(csv_filepath):
    """
    Hàm đọc một file CSV và tạo ra một file JSON tương ứng
    với cùng tên nhưng khác phần mở rộng.
    """
    # Lấy tên file gốc không có phần mở rộng (ví dụ: 'english_grammar_rules')
    base_name = os.path.splitext(csv_filepath)[0]
    json_filepath = f"{base_name}.json"

    data = []
    try:
        # Mở file CSV với encoding='utf-8'
        with open(csv_filepath, mode='r', encoding='utf-8') as csv_file:
            # DictReader tự động lấy dòng đầu tiên làm key
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        print(f"✅ Đọc thành công file: {csv_filepath}")

        # Ghi dữ liệu đã đọc ra file JSON
        with open(json_filepath, 'w', encoding='utf-8') as json_file:
            # ensure_ascii=False để giữ ký tự Unicode (phiên âm, tiếng Việt)
            # indent=4 để file JSON đẹp và dễ đọc
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"🚀 Đã tạo file '{json_filepath}' thành công!")

    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{csv_filepath}'. Vui lòng kiểm tra lại.")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi xử lý file {csv_filepath}: {e}")

# --- Danh sách các file CSV cần chuyển đổi ---
csv_files_to_convert = [
    'english_grammar_rules.csv',
    'english_idioms.csv',
    'english_vocabulary.csv'
]

# --- Vòng lặp để xử lý từng file ---
for file in csv_files_to_convert:
    convert_csv_to_json(file)

print("\n🎉 Hoàn tất quá trình chuyển đổi!")