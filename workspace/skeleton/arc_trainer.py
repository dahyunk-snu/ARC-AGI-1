import os
import json
import pandas as pd
import random

from arc import ARCSolver

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
        N = len(data)
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

def main():
    data_path = "../dataset"
    # aug_data_path = "../dataset_generated"

    use_pretrained_model = False

    train_dataset = load_data(data_path)

    token = os.environ.get("HF_TOKEN", None)
    solver = ARCSolver(token=token)
    solver.train(train_dataset, pretrained=use_pretrained_model)


if __name__ == "__main__":
    main()