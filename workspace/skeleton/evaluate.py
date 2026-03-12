import unsloth #Unsloth should be imported before transformers to ensure 
               # all optimizations are applied. 
               # Your code may run slower or encounter memory issues 
               # without these optimizations.
import numpy as np
from tqdm.auto import tqdm
import os

from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import json

def check_match(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)

    if len(pred.shape) != 2 or pred.shape != truth.shape:
        return 0
    else:
        return int(np.all(pred == truth))

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
    MAX_LEN = 1000
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


def main():
    token = os.environ.get("HF_TOKEN", None)
    from arc import ARCSolver

    solver = ARCSolver(token=token)

    # optional: use adapter_path when you want to put this on the base model.
    # You already merge the model, dismiss it.
    # adapter_path = "artifacts/checkpoint-final-test"
    solver.prepare_evaluation()

    set_seed(1234567890)
    # data_path = "/home/student/subin_workspace/2025DL-team-project/workspace/evaluation_data"
    data_path = "/workspace/dataset"
    N_data = 50

    scores = []
    df = load_data(data_path)

    from datasets import Dataset
    eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))
    for eval_data in tqdm(eval_dataset):
        preds = solver.predict(
            eval_data["train"],
            eval_data["test"][0]["input"],
        )
        s = check_match(preds, eval_data["test"][0]["output"])
        scores.append(s)
        # print if the prediction is correct
        if s == 1:
            print(f"Correct prediction for task {eval_data['task']}")
        else:
            print(f"Incorrect prediction for task {eval_data['task']}")
        # print the numbr of correct predictions above totla predictions
        print(f"Correct predictions: {sum(scores)}/{len(scores)}")
    
    score = np.array(scores).mean() * 100
    print(f"Evaluation scores: {score:.2f}", flush=True)
    print("Evaluation Success")


if __name__ == "__main__":
    main()
