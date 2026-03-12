import numpy as np
from tqdm.auto import tqdm
import os
import json
from arc import ARCSolver

def load_data(base_dir, aug_dir = None):
    """
    Load all ARC tasks from JSON files in base_dir.
    Returns:
        List[List[Dict]]: dataset where
            - len(dataset) == number of .json files (tasks)
            - for each i, len(dataset[i]) == number of examples in that task
    """

    filenames = [f for f in os.listdir(base_dir) if f.endswith('.json')]
    data_files = [os.path.join(base_dir, fn) for fn in filenames]

    dataset = []
    for fn in data_files:
        with open(fn, 'r') as fp:
            data = json.load(fp)
        dataset.append(data)

    if aug_dir is not None:
        aug_filenames = [f for f in os.listdir(aug_dir) if f.endswith('.json')]
        aug_data_files = [os.path.join(aug_dir, fn) for fn in filenames]

        for fn in aug_data_files:
            with open(fn, 'r') as fp:
                data = json.load(fp)
            dataset.append(data)

    return dataset

def build_train_dataset(raw_tasks, start = 0):
    """
    From raw_tasks, build few-shot training entries.
    Args:
        raw_tasks: List of tasks, each task is a list of examples (dicts)
        shot: number of demos per example
    Returns:
        List[Dict]: each dict has 'train' (List of demos) and 'test' (single example)
    """

    fewshot = []

    N = len(raw_tasks)

    MAX_LEN = 10000 + start
    pbar = tqdm(total=MAX_LEN, desc="Building few-shot entries", unit="entry")
    rng = np.random.default_rng(42)

    while len(fewshot) < MAX_LEN:
        task_idx = rng.integers(0, N)
        task = raw_tasks[task_idx]

        n_task = len(task)
        grids_idx =  rng.choice(n_task, size=4, replace=True)
        train_grids = [task[i] for i in grids_idx[:3]]
        test_grids = [task[i] for i in grids_idx[3:]]

        fewshot.append({
            'train': train_grids,
            'test': test_grids
        })
        pbar.update(1)

    pbar.close()
    return fewshot[start:]


def main():
    data_path = "../dataset"
    aug_data_path = "../dataset_generated"

    # start = int(input("\nEnter the start number of the training set: \n"))
    start = 0
    use_pretrained_model = False

    while(True):
        # include_aug_data = input("\nEnter Y/N whether to include the augmented training set: \n"):
        include_aug_data = "N"
        if "Y" in include_aug_data or "y" in include_aug_data:
            raw_data = load_data(data_path, aug_dir=aug_data_path)
            break
        elif "N" in include_aug_data or "n" in include_aug_data:
            raw_data = load_data(data_path)
            break
        else:
            print("Invalid input. Please try again.\n")

    train_dataset = build_train_dataset(raw_data, start=start)

    token = os.environ.get("HF_TOKEN", None)
    solver = ARCSolver(token=token)
    solver.train(train_dataset, pretrained=use_pretrained_model)


if __name__ == "__main__":
    main()