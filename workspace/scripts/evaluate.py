import numpy as np
from tqdm.auto import tqdm

from transformers import set_seed
from datasets import load_dataset

def check_match(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)

    if len(pred.shape) != 2 or pred.shape != truth.shape:
        return 0
    else:
        return int(np.all(pred == truth))


def main():
    from arc import ARCSolver

    solver = ARCSolver()
    solver.prepare_evaluation()

    set_seed(1234567890)

    data_path = "dataset/eval/data.json"
    N_data = 10

    scores = []
    eval_dataset = load_dataset('json', data_files={
        "eval": data_path
        }, split='eval').shuffle(42).select(range(N_data))
    # dataset = load_dataset('json', data_files=data_path, split='train')
    # dataset = dataset.shuffle(42).train_test_split(test_size=0.2)
    # dataset = 
    for eval_data in tqdm(eval_dataset):
        preds = solver.predict(
            eval_data["train"],
            eval_data["test"][0]["input"],
        )
        s = check_match(preds, eval_data["test"][0]["output"])
        scores.append(s)

    
    score = np.array(scores).mean() * 100
    print(f"Evaluation scores: {score:.2f}", flush=True)

if __name__ == "__main__":
    main()
