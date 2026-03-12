import json
import os
from pathlib import Path

def convert_jsonl_to_json(input_dir: str, output_dir: str):
    """
    input_dir 폴더 내의 모든 .jsonl 파일을 읽어서,
    각 줄(task)마다 examples 필드의 input-output 페어를 추출하여
    [{"input": […], "output": […]}] 형식의 새로운 .json 파일로 저장합니다.
    배열(list)은 모두 한 줄에 출력됩니다.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # input_dir 내의 모든 .jsonl 파일 순회
    for jsonl_file in input_path.glob("*.jsonl"):
        with jsonl_file.open('r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr):
                task = json.loads(line)  # 각 줄을 JSON으로 파싱
                examples = task.get("examples", [])

                # examples에서 [input, output] 페어를 {"input":…, "output":…} 로 변환
                converted = []
                for input_grid, output_grid in examples:
                    converted.append({
                        "input": input_grid,
                        "output": output_grid
                    })

                # 출력 파일명: 원본 파일명_라인번호.json (예: taskfile_0.json)
                base_name = jsonl_file.stem
                out_filename = f"{base_name}_{idx}.json"
                out_file = output_path / out_filename

                # compact 모드: 배열을 모두 한 줄로 출력
                with out_file.open('w', encoding='utf-8') as fw:
                    json.dump(
                        converted,
                        fw,
                        ensure_ascii=False,
                        separators=(', ', ': ')
                    )

    print(f"변환 완료! '{input_dir}' → '{output_dir}'")

if __name__ == "__main__":
    # 실제로 사용하실 때에는 아래 두 경로를 알맞게 바꿔주세요.
    convert_jsonl_to_json(
        input_dir="/home/student/subin_workspace/2025DL-team-project/workspace/BARC/generated_problems",
        output_dir="/home/student/subin_workspace/2025DL-team-project/workspace/BARC/dataset_generated"
    )