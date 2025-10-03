import json
import os

# Define the file path
file_path = './ocr_data_result/final_result.json'


def load_json_from_file_with_duplicates(file_path):
    """
    從檔案路徑讀取 JSON 資料，並使用 object_pairs_hook=list 參數以保留重複的鍵。
    """
    try:
        # 使用 'with open' 確保檔案在讀取完成後會被自動關閉
        # 'r' 表示讀取模式，encoding='utf-8' 確保能正確處理中文字元
        with open(file_path, 'r', encoding='utf-8') as file:
            # 關鍵：使用 json.load() (而不是 loads)，直接從檔案物件讀取
            # object_pairs_hook=list 確保 JSON 物件轉換為 List of Tuples
            data = json.load(file, object_pairs_hook=list)
            return data
            
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認檔案是否存在於當前目錄。")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{file_path}' 的內容不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return None

# 執行讀取
raw_data = load_json_from_file_with_duplicates(file_path)

# --- 處理與顯示結果 ---
if raw_data:
    print(f"--- 成功從檔案 '{file_path}' 讀取資料 ---")
    print(f"外層資料類型: {type(raw_data)}")
    print("-" * 40)

    # 遍歷外層資料：(file_name, inner_list)
    for file_name, inner_list in raw_data:
        print(f"檔案名稱 (外層 Key): {file_name}")
        
        # 遍歷內層清單：(key, value)，即 ID 和 Location
        print("  內層重複鍵值對清單:")
        for key, value in inner_list:
            print(f"    -> ID: {key}, Location: {value}")
        
        print("-" * 40)