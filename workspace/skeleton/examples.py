from utils import render_grid
from rich import print
import os
import pandas as pd
from typing import List
import json
import numpy as np


def load_data(base_dir):
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)
        dataset.append(data)

    filenames = [fn.split(".")[0] for fn in filenames]
    data = []
    MAX_LEN = 10
    rng = np.random.default_rng(42)

    N = len(dataset)

    while len(data) < MAX_LEN:
        task_idx = rng.integers(0, N)
        task = dataset[task_idx]
        file_name = filenames[task_idx]

        n_task = len(task)
        grids_idx =  rng.choice(n_task, size=4, replace=True)
        train_grids = [task[i] for i in grids_idx[:3]]
        test_grids = [task[i] for i in grids_idx[3:]]

        test_inputs = [{'input': grid['input']} for grid in test_grids]
        test_outputs = [grid['output'] for grid in test_grids]
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    df = pd.DataFrame(data)
    return df


N_data = 10
data_path = "/workspace/dataset"
df = load_data(data_path)

from datasets import Dataset
dataset = Dataset.from_pandas(df).shuffle(41).select(range(N_data))

print("-----Dataset Statistics-----")
print(dataset)
print("-----Train Question Example-----")
print(dataset[0]['train'])
print("-----Test Question Example-----")
print(dataset[0]['test'])

for i in range(len(dataset)):
    print("Train Input")
    render_grid(dataset[i]['train'][0]['input'])
    print("Train Output")
    render_grid(dataset[i]['train'][0]['output'])

    print("Test Input")
    render_grid(dataset[i]['test'][0]['input'])
    print("Test Output")
    render_grid(dataset[i]['test'][0]['output'])
