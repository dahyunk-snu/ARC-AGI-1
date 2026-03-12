from .task import Task

#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict
import warnings

# 경고 필터링: "valid feature names" 관련된 UserWarning 무시
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)

def cluster_tasks(task_list: List[Task], n_clusters: int = 10,random_state=42,model=None,scalar=None) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    task_list 안의 Task 인스턴스들을 feature-based로 클러스터링 합니다.

    Args:
        task_list: Task 인스턴스 리스트 (각 task.feat에 대표 피처 dict가 저장되어 있어야 함)
        n_clusters: 생성할 클러스터 수

    Returns:
        df: task 이름을 인덱스로, 원본 피처와 cluster 라벨을 가진 DataFrame
        kmeans: 학습된 KMeans 모델
    """
    # 1) DataFrame 생성
    data = {}
    for task in task_list:
        # 피처가 계산되지 않았다면 추출
        if task.feat is None:
            task.extract_features()
        data[task.name] = task.feat
    df = pd.DataFrame.from_dict(data, orient='index')

    # 2) 스케일링
    if scalar is None:
        scaler = StandardScaler()
    else:
        print("Using provided scaler")
        scaler = scalar
    X = scaler.fit_transform(df)

    # 3) KMeans 클러스터링
    if model is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        print("Using provided model")
        kmeans = model
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels

    # 4) 실루엣 점수 평가
    sil_score = silhouette_score(X, labels)
    print(f"Silhouette Score (k={n_clusters}): {sil_score:.3f}")

    # 5) 클러스터별 task 목록 출력
    for c in sorted(df['cluster'].unique()):
        members = df[df['cluster'] == c].index.tolist()
        print(f"Cluster {c} ({len(members)} tasks): {members}")

    return df, kmeans, scaler


def single_sample_silhouette(sample: np.ndarray, kmeans: KMeans, label = None) :
    """
    한 샘플에 대해 silhouette-like score 계산.

    Args:
        sample: shape (1, n_features) – 이미 scaler.transform 된 데이터
        kmeans: 학습된 KMeans 모델 (fitted)

    Returns:
        s: silhouette-like score (float)
    """
    # 1) 할당된 클러스터 라벨
    if label is None:
        label = kmeans.predict(sample)[0]

    # 2) 중심(centroids) 불러오기
    centers = kmeans.cluster_centers_

    # 3) a: 자기 클러스터 중심과의 거리
    a = np.linalg.norm(sample - centers[label])

    # 4) b: 다른 모든 중심들과의 거리 중 최소값
    #    (자기 중심은 제외)
    other_centers = np.delete(centers, label, axis=0)
    b = np.min(np.linalg.norm(sample - other_centers, axis=1))

    # 5) silhouette-like 계산
    return (b - a) / max(a, b), label

