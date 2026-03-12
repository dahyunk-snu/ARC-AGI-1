import os
import json
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
from rich.console import Console
from rich.text import Text

# JSON 파일명만 모아 리스트로 저장

def list_json_files(dataset_dir_list):
    # 디렉토리 내 파일 중 .json 확장자를 가진 파일만 필터링
    return [
        dataset_dir + '/'+f[:-5]
        for dataset_dir in dataset_dir_list
        for f in os.listdir(dataset_dir)
        if f.endswith('.json')
    ]
def compressive_activation(x, a,b, beta=10.0, normalize: bool = True, eps=1e-6):
    """
    1) x <= a : ≈0,
    2) a < x < b : ≈(x - a),
    3) x >= b : ≈(b - a),
    모두 Softplus(beta)로 부드럽게 연결합니다.

    Args:
        x: Tensor
        a, b: scalars or tensors, a < b
        beta: Softplus 스무딩 계수 (크면 더 Sharp)
        normalize: True면 최댓값(b-a)으로 나눠 [0,1]로 정규화
    """
    # 부드러운 ReLU: softplus(z) ≈ log(1 + exp(beta*z)) / beta
    # 차분을 취하면 [0→x−a→b−a] 부드럽게 얻음
    # x = torch.tensor(x)
    # s1 = F.softplus(x - a, beta=beta)
    # s2 = F.softplus(x - b, beta=beta)
    # out = (s1 - s2)
    # if normalize:
    #     out = out / ((b - a) + eps)
    # return out

    # version 2

    return np.tanh(2*(x-(a+b)/2)/(b-a)) * (b-a)/2 + (a+b)/2
# 시각화에 사용할 색상 매핑 및 콘솔 설정
color_map = {
    0: "black", 1: "red", 2: "green", 3: "yellow", 4: "blue",
    5: "magenta", 6: "cyan", 7: "white", 8: "bright_red", 9: "bright_green",
}
console = Console()

def make_rich_lines(grid: List[List[int]], cell_width: int = 4) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append(" " * cell_width, style=f"on {color}")
        lines.append(visual)
    return lines

def render_grids_parallel(grids: List[List[List[int]]], cell_width: int = 4, spacing: int = 4):
    rich_grids = [make_rich_lines(grid, cell_width) for grid in grids]
    max_rows = max(len(lines) for lines in rich_grids)
    for lines in rich_grids:
        while len(lines) < max_rows:
            lines.append(Text(" " * len(lines[0])))
    for row_index in range(max_rows):
        combined = Text()
        for lines in rich_grids:
            combined += lines[row_index]
            combined += Text(" " * spacing)
        console.print(combined)

class FeatureCalculator:
    @staticmethod
    def area(grid: List[List[int]]) -> int:
        # 전체 셀 개수
        return len(grid) * len(grid[0]) if grid and grid[0] else 0

    @staticmethod
    def unique_colors(grid: List[List[int]]) -> set:
        # 그리드 내 고유 색상 집합
        return set(np.array(grid).flatten().tolist())

    @staticmethod
    def removed_color_ratio(inp: List[List[int]], out: List[List[int]]) -> float:
        # 입력 색상 중 출력에 없는 색상 비율
        in_colors = FeatureCalculator.unique_colors(inp)
        out_colors = FeatureCalculator.unique_colors(out)
        if not in_colors:
            return 0.0
        removed = in_colors.difference(out_colors)
        return len(removed) / len(in_colors)


    @staticmethod
    def freq_ratio(grid: List[List[int]]) -> float:
        # 가장 빈도 높은 색상 / 전체 픽셀 수
        flat = np.array(grid).flatten().tolist()
        counts = Counter(flat)
        if not counts:
            return 0.0
        total = len(flat)
        top = counts.most_common(1)[0][1]
        return top / total if total > 0 else 0.0

    @staticmethod
    def changed_ratio(inp: List[List[int]], out: List[List[int]]) -> float:
        # 변경된 셀 수 / max(입력셀수, 출력셀수)
        arr_in = np.array(inp)
        arr_out = np.array(out)
        # 겹치는 영역에서 차이 개수
        min_h = min(arr_in.shape[0], arr_out.shape[0])
        min_w = min(arr_in.shape[1], arr_out.shape[1])
        diff = (arr_in[:min_h, :min_w] != arr_out[:min_h, :min_w]).sum()
        # 크기 차이만큼 추가 차이
        extra = abs(arr_in.size - arr_out.size)
        total_diff = diff + extra
        denom = max(arr_in.size, arr_out.size)
        return total_diff / denom if denom > 0 else 0.0

    @staticmethod
    def complexity_2x2(grid: List[List[int]]) -> int:
        # 2x2 패치 기반 고유 서브그리드 개수, 크기 미만이면 0
        arr = np.array(grid)
        h, w = arr.shape
        if h < 2 or w < 2:
            return 0
        patches = set()
        for i in range(h - 1):
            for j in range(w - 1):
                patch = tuple(arr[i:i+2, j:j+2].flatten().tolist())
                patches.add(patch)
        return len(patches)

    @staticmethod
    def same_color_edge_ratio(grid: List[List[int]]) -> float:
        arr = np.array(grid)
        h, w = arr.shape
        same = 0
        total = 0
        for i in range(h):
            for j in range(w - 1):
                total += 1
                if arr[i, j] == arr[i, j + 1]:
                    same += 1
        for i in range(h - 1):
            for j in range(w):
                total += 1
                if arr[i, j] == arr[i + 1, j]:
                    same += 1
        return same / total if total > 0 else 0.0

    @staticmethod
    def color_diversity_2x2(grid: List[List[int]]) -> float:
        """
        모든 2x2 블록에서 고유 색상 개수를 계산하여 그 평균을 반환.
        크기가 2보다 작으면 0.0 반환.
        """
        arr = np.array(grid)
        h, w = arr.shape
        if h < 2 or w < 2:
            return 0.0
        diversities = []
        for i in range(h - 1):
            for j in range(w - 1):
                block = arr[i:i+2, j:j+2]
                diversities.append(len(set(block.flatten().tolist())))
        return float(np.mean(diversities)) if diversities else 0.0
    
class Task:
    """
    Represents a single ARC task, loading input-output examples automatically
    from a JSON file given only its task name.

    Attributes:
        name (str): Task name (JSON filename without extension).
        examples (List[Tuple[List[List[int]], List[List[int]]]]):
            List of (input_grid, output_grid) pairs.
    """
    def __init__(self, name: str):
        self.name = name
        self.examples = []
    def input_example(self, example: dict):
        self.examples.append(example)


    def load_examples(self) -> List[Dict[str, List[List[int]]]]:
        """
        내부 JSON 파일(dataset_dir/{name}.json)을 읽어들여
        examples 리스트를 반환합니다.
        """
        filename = f"{self.name}.json"
        filepath = os.path.join(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Task file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        examples: List[Dict[str, List[List[int]]]] = []
        for entry in data:
            examples.append({
                'input': entry['input'],    # type: List[List[int]]
                'output': entry['output'],  # type: List[List[int]]
            })

        self.examples = examples
        return examples
    def extract_features(self):
        """
        모든 예제에 대해 개별 피처 계산 후 평균을 구해 대표 벡터로 반환.
        반환 딕셔너리 키는 피처 이름.
        """
        feats_list = []
        for item in self.examples:
            inp = item['input']
            out = item['output']
            f = {
                'area_ratio': FeatureCalculator.area(out) / FeatureCalculator.area(inp) if FeatureCalculator.area(inp) > 0 else 0.0,
                'input_color_count': len(FeatureCalculator.unique_colors(inp)),
                'removed_color_ratio': FeatureCalculator.removed_color_ratio(inp, out),
                'freq_ratio': FeatureCalculator.freq_ratio(inp),
                'changed_ratio': FeatureCalculator.changed_ratio(inp, out),
                # 'complexity_2x2': FeatureCalculator.complexity_2x2(inp),
                'same_color_edge_ratio': FeatureCalculator.same_color_edge_ratio(inp),
                'color_diversity_2x2': FeatureCalculator.color_diversity_2x2(inp),
            }
            feats_list.append(f)
        self.feats_list = feats_list
    def representative_features(self) -> Dict[str, float]:
        # 각 피처별 평균 계산
        feats_list = self.feats_list
        agg = {}
        for key in feats_list[0].keys():
            values = [d[key] for d in feats_list]
            #################  Handle here #################
            if key in ['area_ratio']:
                agg[key + '_representative'] = float(np.tanh(2*np.mean(np.log(values)))) # +1, 0, -1로 쏠림
            # if key in ['input_color_count']:
                # agg[key + '_representative'] = float(compressive_activation(np.mean(values),1.5,5.5)) #  activation
            if key in ['changed_ratio']:
                agg[key + '_representative'] = float(compressive_activation(np.mean(values),0.3,0.9)) # activation
            if key in ['freq_ratio']:
                agg[key + '_representative'] = float(compressive_activation(np.mean(values),0.5,0.95))
            # if key in ['removed_color_ratio']:
                # agg[key + '_representative'] = float(compressive_activation(np.mean(values),0.2,0.45)) # activation
            # if key in ['color_diversity_2x2']:
                # agg[key + '_representative'] = float(compressive_activation(np.exp(np.mean(np.log(values))),1.5,2.5)) # 기하평균 activation
            # if key in ['same_color_edge_ratio']:
                # agg[key + '_representative'] = float(compressive_activation(np.mean(values),0.1,0.7)) # activation
        self.feat = agg
        return agg
    def show_examples(self, n: int = 2):
        """최대 n개의 input-output pair를 시각화 출력"""
        for i, item in enumerate(self.examples[:n]):
            inp = item['input']
            out = item['output']
            print(f"Example {i+1} of Task {self.name}")
            render_grids_parallel([inp, out], cell_width=3, spacing=4)
    def classify_task(self,model_dir) -> tuple:
        from .task_kmean import single_sample_silhouette
        with open(model_dir + '/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open(model_dir + '/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        df_ex = pd.DataFrame.from_dict(self.feat, orient='index')
        X_ex = scaler.transform(df_ex[0].values.reshape(1,-1))
        s, label = single_sample_silhouette(X_ex,model)
        self.label = label
        self.label_score = s
        return label,s
    def __repr__(self):
        return f"<Task name={self.name!r}, examples={len(self.examples)} pairs>"
