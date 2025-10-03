import json
import os
import csv

# Define file paths
FINAL_RESULT_PATH = './ocr_data_result/final_result.json'
OCR_ANSWER_PATH = './ocr_data_result/ocr_ans.csv'

def load_json_from_file_with_duplicates(file_path):
    """
    從檔案路徑讀取 JSON 資料，並使用 object_pairs_hook=list 參數以保留重複的鍵。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

def load_csv_ground_truth(file_path):
    """
    從 CSV 檔案讀取 Ground Truth 資料。
    預期 CSV 格式: FileName,PartNo,COO
    """
    ground_truth = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                file_name = row['FileName']
                part_no = row['PartNo']
                coo = row['COO']
                if file_name not in ground_truth:
                    ground_truth[file_name] = []
                ground_truth[file_name].append((part_no, coo))
        return ground_truth
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。請確認檔案是否存在於當前目錄。")
        return None
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return None

def normalize_coo(coo_value):
    """
    標準化 COO 值，將常見的縮寫和大小寫不一致的國家名稱統一。
    """
    if not isinstance(coo_value, str):
        return str(coo_value).upper().strip()

    coo_value = coo_value.upper().strip()
    
    # 建立一個映射表，將常見的縮寫和變體映射到標準名稱
    coo_map = {
        "SG": "SINGAPORE",
        "TW": "TAIWAN",
        "TH": "THAILAND",
        "MY": "MALAYSIA",
        "VN": "VIETNAM",
        "CN": "CHINA",
        "US": "USA",
        "USA": "USA",
        "INDONESIA": "INDONESIA",
        "VIETNAM": "VIETNAM",
        "MALAYSIA": "MALAYSIA",
        "SINGAPORE": "SINGAPORE",
        "TAIWAN": "TAIWAN",
        "CHINA": "CHINA",
        "THAILAND": "THAILAND",
        "未提供": "UNKNOWN" # Handle '未提供' as a specific unknown
    }
    return coo_map.get(coo_value, coo_value) # Return original if not in map

def calculate_precision_recall(ocr_results, ground_truth):
    """
    計算 OCR 結果的 Precision 和 Recall。
    只針對有重疊的 FileName 和 PartNo 進行計算。
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    results_by_file = {}

    for file_name_ocr, ocr_data_list in ocr_results:
        file_name_ocr_str = str(file_name_ocr) # Ensure file_name is string for comparison
        if file_name_ocr_str in ground_truth:
            gt_data_list = ground_truth[file_name_ocr_str]

            # Convert lists of tuples to sets of normalized (PartNo, COO) for easier comparison
            # Handle potential duplicate keys in OCR results by treating each (PartNo, COO) as a distinct prediction
            ocr_pairs = set()
            for part_no, coo in ocr_data_list:
                ocr_pairs.add((str(part_no), normalize_coo(coo)))
            
            gt_pairs = set()
            for part_no, coo in gt_data_list:
                gt_pairs.add((str(part_no), normalize_coo(coo)))

            # Filter for overlapping PartNo keys
            ocr_part_nos = {pair[0] for pair in ocr_pairs}
            gt_part_nos = {pair[0] for pair in gt_pairs}
            
            overlapping_part_nos = ocr_part_nos.intersection(gt_part_nos)

            # Filter ocr_pairs and gt_pairs to only include overlapping PartNo keys
            filtered_ocr_pairs = {pair for pair in ocr_pairs if pair[0] in overlapping_part_nos}
            filtered_gt_pairs = {pair for pair in gt_pairs if pair[0] in overlapping_part_nos}

            true_positives = len(filtered_ocr_pairs.intersection(filtered_gt_pairs))
            false_positives = len(filtered_ocr_pairs.difference(filtered_gt_pairs))
            false_negatives = len(filtered_gt_pairs.difference(filtered_ocr_pairs))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            results_by_file[file_name_ocr_str] = {
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
        else:
            results_by_file[file_name_ocr_str] = {
                "precision": 0,
                "recall": 0,
                "true_positives": 0,
                "false_positives": len(ocr_data_list), # All OCR results are FPs if no GT
                "false_negatives": 0
            }


    overall_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    return results_by_file, overall_precision, overall_recall

# Load data
ocr_data_raw = load_json_from_file_with_duplicates(FINAL_RESULT_PATH)
ground_truth_data = load_csv_ground_truth(OCR_ANSWER_PATH)

if ocr_data_raw and ground_truth_data:
    # Convert raw OCR data (list of tuples) to a dictionary for easier lookup by file_name
    ocr_results_dict = {}
    for file_name, inner_list in ocr_data_raw:
        ocr_results_dict[str(file_name)] = inner_list

    # The calculate_precision_recall expects a list of (file_name, data_list)
    # so we need to convert ocr_results_dict back to that format for the function call
    ocr_results_for_calc = [(k, v) for k, v in ocr_results_dict.items()]

    file_metrics, overall_precision, overall_recall = calculate_precision_recall(ocr_results_for_calc, ground_truth_data)

    print("--- Precision and Recall Results ---")
    for file_name, metrics in file_metrics.items():
        print(f"File: {file_name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print("-" * 30)
    
    print(f"\nOverall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
else:
    print("無法載入 OCR 結果或 Ground Truth 資料，無法計算指標。")
