import os
import json
import pandas as pd
import random
import arc

from arc import ARCSolver
from arc.task_clusterization.clusterization import save_cluster_model, classify_cluster


def load_data(base_dir):
    print("\n▶ Loading training datasets\n")

    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)
        dataset.append(data)

    filenames = [fn.split(".")[0] for fn in filenames]
    arc_data = []
    print(sum([len(data) for data in dataset]))
    for task_idx, data in enumerate(dataset):
        random.shuffle(data)
        N = 500#len(data)
        for i in range(0, N, 4):
            if i + 3 >= N:
                continue
            train_grids = [grid for grid in data[i:i+3]]
            test_grids = [grid for grid in data[i+3:i+4]]

            arc_data.append({
            'train': train_grids,
            'test': test_grids
        })

    random.seed(42) 
    random.shuffle(arc_data)
    print("\n▶ Successfully loaded training datasets: ", len(arc_data), "\n")
    return arc_data
def split_data(dataset, cluster_model_dir: str, n_clusters: int):
    """
    주어진 dataset을 classify_cluster 결과에 따라 n_clusters개의 그룹으로 분할합니다.

    Args:
        dataset (list of dict): 각 요소가 적어도 'train' 키를 가진 dict로, 분류할 데이터 예시를 포함합니다.
        cluster_model_dir (str): classify_cluster 함수가 사용할 모델 디렉터리 경로입니다.
        n_clusters (int): 클러스터 개수 (0부터 n_clusters-1 까지 레이블이 나타납니다).

    Returns:
        List[List[dict]]: i번째 리스트에는 레이블 i로 분류된 데이터 포인트들을 담고 있습니다.
    """
    # 레이블별로 데이터를 담을 빈 리스트 초기화
    clusters = [[] for _ in range(n_clusters)]

    for idx, item in enumerate(dataset):
        # 데이터 예시는 'train' 키 아래에 있다고 가정
        train_example = item['train']
        label,_ = classify_cluster(cluster_model_dir, train_example)

        # 레이블 유효성 검사
        if not (0 <= label < n_clusters):
            raise ValueError(f"인덱스 {idx}의 예제에 대해 잘못된 레이블이 반환되었습니다: {label}")

        clusters[label].append(item)

    return clusters

def main():
    data_path = "../dataset_final"
    # aug_data_path = "../dataset_generated"

    use_pretrained_model = True

    train_dataset = load_data(data_path)
    # train_dataset = train_dataset[:50] # For testing
    DATASET_DIR_LIST = [data_path]
    cluster_model_dir = "artifacts/cluster_model_fixed"
    n_clusters = 6
    # save_cluster_model(DATASET_DIR_LIST, cluster_model_dir, n_clusters)
    train_dataset_splitted = split_data(train_dataset,cluster_model_dir,n_clusters)
    for i in range(len(train_dataset_splitted)):
        print(f"Cluster {i} has {len(train_dataset_splitted[i])} items")
    token = os.environ.get("HF_TOKEN", None)
    for classes in [[1],[2],[5]]:
        solver = ARCSolver(token=token)
        # solver.train(train_dataset_splitted, pretrained=use_pretrained_model,classes=classes)
        del solver

if __name__ == "__main__":
    main()