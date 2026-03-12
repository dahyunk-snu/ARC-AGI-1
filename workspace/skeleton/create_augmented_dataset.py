import os
import json
import numpy as np
import os
import random
from tqdm.auto import tqdm

def rotate_grid(grid, k=1):
    """
    Rotate a 2D grid by 90 degrees k times.
    Args:
        grid: List of List of ints
        k: number of 90° rotations (clockwise)
    Returns:
        rotated grid as List of List of ints
    """
    arr = np.array(grid)
    rotated = np.rot90(arr, -k)
    return rotated.tolist()


def transpose_grid(grid, main=True):
    """
    Transpose a 2D grid by main diagonal
    Args:
        grid: List of List of ints
    Returns:
        transposed grid as List of List of ints
    """
    arr = np.array(grid)
    transposed = np.swapaxes(arr, 0, 1)
    if not main:
        transposed = np.rot90(transposed, 2)
    return transposed.tolist()


def reflect_grid_x(grid):
    """
    Transpose a 2D grid by x axis
    Args:
        grid: List of List of ints
    Returns:
        reflected grid as List of List of ints
    """
    arr = np.array(grid)
    reflected = arr[::-1]
    return reflected.tolist()


def reflect_grid_y(grid):
    """
    Transpose a 2D grid by x axis
    Args:
        grid: List of List of ints
    Returns:
        reflected grid as List of List of ints
    """
    arr = np.array(grid)
    reflected = np.fliplr(arr) 
    return reflected.tolist()



def permute_values(grid, mapping):
    """
    Permute cell values according to a mapping dict.
    Args:
        grid: List of List of ints
        mapping: dict {old_value: new_value}
    Returns:
        permuted grid
    """
    arr = np.array(grid)
    permuted = np.asarray(mapping)[arr]
    return permuted
    # return [[mapping.get(cell, cell) for cell in row] for row in grid]


def generate_random_mapping():
    """
    Create a random permutation mapping for given pixel values.
    """
    
    values = list(range(10))
    rng = np.random.default_rng(42)
    perm = rng.permutation(values).tolist()
    return perm


def swap_io(ex):
    """
    Swap input and output of an example if shapes match.
    Args:
        ex: dict with 'input' and 'output'
    Returns:
        swapped example or None
    """

    return {'input': ex['output'], 'output': ex['input']}


def main():
    PROPERTY_PATH = 'task_property.json'
    DATASET_DIR = '../dataset'
    AUGMENTED_DIR = '../augmented_dataset'

    # Make directory to store augmented data
    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    # Load task property
    with open(PROPERTY_PATH, 'r') as f:
        task_props = json.load(f)

    # Iterate tasks
    for task_id, props in tqdm(task_props.items()):
        file_path = os.path.join(DATASET_DIR, f"{task_id}.json")
        if not os.path.exists(file_path):
            print(f"File missing: {file_path}")
            continue

        with open(file_path, 'r') as f:
            samples = json.load(f)
        
        if len(samples) < 500:
            continue

        can_rotate = props.get("rotate", False)
        can_transpose = props.get("transpose", False)
        can_permute = props.get("permute_color", False)
        keep_background = props.get("keep_background", False)

        # 1. Rotation (0, 90, 180, 270 deg)
        for k in range(4):
            rotated_samples = []
            for pair in samples:
                new_input = rotate_grid(pair['input'], k=k)
                new_output = rotate_grid(pair['output'], k=k)
                rotated_samples.append({"input": new_input, "output": new_output})

            if can_rotate or k == 0:
                save_path = os.path.join(AUGMENTED_DIR, f"{task_id}.json")
            else:
                save_path = os.path.join(AUGMENTED_DIR, f"{task_id}_rt{k}.json")

            with open(save_path, 'w') as f:
                json.dump(rotated_samples, f)


        # 2. Transpose main diagonal
        transposed_samples = []
        for pair in samples:
            new_input = transpose_grid(pair['input'])
            new_output = transpose_grid(pair['output'])
            transposed_samples.append({"input": new_input, "output": new_output})

        if can_transpose:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}.json")
        else:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}_tp1.json")
        with open(save_path, 'w') as f:
            json.dump(transposed_samples, f)
            
        # 3. Transpose sub diagonal
        transposed_samples = []
        for pair in samples:
            new_input = transpose_grid(pair['input'], False)
            new_output = transpose_grid(pair['output'], False)
            transposed_samples.append({"input": new_input, "output": new_output})

        if can_transpose:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}.json")
        else:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}_tp2.json")
        with open(save_path, 'w') as f:
            json.dump(transposed_samples, f)
            
        # 4. Reflect x axis
        reflected_samples = []
        for pair in samples:
            new_input = reflect_grid_x(pair['input'])
            new_output = reflect_grid_x(pair['output'])
            reflected_samples.append({"input": new_input, "output": new_output})

        if can_rotate and can_transpose:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}.json")
        else:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}_x.json")
        with open(save_path, 'w') as f:
            json.dump(reflected_samples, f)
            
        # 5. Reflect y axis
        reflected_samples = []
        for pair in samples:
            new_input = reflect_grid_y(pair['input'])
            new_output = reflect_grid_y(pair['output'])
            reflected_samples.append({"input": new_input, "output": new_output})

        if can_rotate and can_transpose:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}.json")
        else:
            save_path = os.path.join(AUGMENTED_DIR, f"{task_id}_y.json")
        with open(save_path, 'w') as f:
            json.dump(reflected_samples, f)

    print("Data augmentation finished")


if __name__ == '__main__':
    main()