from .task import list_json_files, Task
from .task_kmean import cluster_tasks, single_sample_silhouette
import pickle
import pandas as pd
import os

def save_cluster_model(DATASET_DIR_LIST, model_dir,n_clusters):
    json_files = list_json_files(DATASET_DIR_LIST)
    # task 객체 리스트 작성
    task_list = [Task(name) for name in json_files]
    for _,task in enumerate(task_list):
        # 각 task 객체에 대해 load_examples, extract_features, representative_features 메소드 호출
        task.load_examples()
        task.extract_features()
        task.representative_features()
    df_clusters, model, scaler = cluster_tasks(task_list, n_clusters=n_clusters)

    os.makedirs(model_dir, exist_ok=True)
    with open(model_dir + '/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        print("Model saved successfully.")
    with open(model_dir + '/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
        print("Scaler saved successfully.")

def classify_cluster(model_dir, train_example: list[dict],print=False) -> tuple:
    with open(model_dir + '/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(model_dir + '/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    task_single = Task('task_tmp')

    for i in range(len(train_example)):
        task_single.input_example(train_example[i])

    task_single.extract_features()
    task_single.representative_features()
    df_ex = pd.DataFrame.from_dict(task_single.feat, orient='index')
    if print:
        task_single.show_examples()
        # print(df_ex[0].values.reshape(1,-1))
    X_ex = scaler.transform(df_ex[0].values.reshape(1,-1))
    s, label = single_sample_silhouette(X_ex,model)
    return label,s