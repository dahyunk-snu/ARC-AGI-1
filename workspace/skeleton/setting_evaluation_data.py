import os
import json
import shutil

def is_within_size(grid, max_rows=10, max_cols=10):
    """그리드가 주어진 크기 이하인지 확인"""
    return len(grid) <= max_rows and all(len(row) <= max_cols for row in grid)

def file_passes_size_check(json_path, max_rows=10, max_cols=10):
    """파일 내 모든 input/output이 조건을 만족하는지 확인"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data['train'] + data['test']:
        if not (is_within_size(item['input'], max_rows, max_cols) and
                is_within_size(item['output'], max_rows, max_cols)):
            return False
    return True

def filter_json_files_by_size(input_dir, output_dir, max_rows=10, max_cols=10):
    """조건을 만족하는 JSON 파일만 output_dir로 복사"""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for fname in files:
        full_path = os.path.join(input_dir, fname)
        if file_passes_size_check(full_path, max_rows, max_cols):
            shutil.copy(full_path, os.path.join(output_dir, fname))
            print(f"[✔] Passed: {fname}")
        else:
            print(f"[✘] Skipped: {fname}")

input_dir = "./workspace/evaluation_data"
output_dir = "./workspace/true_evaluation_data"
#함수 실행 코드 : size 판별하여 파일 걸러내어 output_dir로 복사.
#filter_json_files_by_size(input_dir, output_dir)

##################################################

def convert_to_old_format(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(input_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        # train + test를 하나의 리스트로 합치기
        all_data = data["train"] + data["test"]

        # 새 파일로 저장
        new_path = os.path.join(output_dir, fname)
        with open(new_path, "w") as f:
            json.dump(all_data, f)

        print(f"[✔] Converted: {fname}")

# 함수 실행 코드. : load_data에 집어넣을 수 있도록 json형식 변환환
convert_to_old_format(input_dir, output_dir)