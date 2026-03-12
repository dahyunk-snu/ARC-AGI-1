from datasets import Dataset
from utils import render_grid, render_grids_parallel
from rich import print
import os
import pandas as pd
from typing import List
import json
import numpy as np
import sys
import random
from enum import Enum


class TaskProperty(Enum):
    ROTATE = 'rotate'
    TRANSPOSE = 'transpose'
    PERMUTE_COLOR = 'permute_color'
    KEEP_BACKGROUND = 'keep_background'


TaskPropertyList = [prop.value for prop in TaskProperty]


def load_data(base_dir, task_id=''):
    filenames = os.listdir(base_dir)
    target_filename = str(task_id) + ".json"
    data_files = [os.path.join(base_dir, p)
                  for p in filenames if target_filename in p]

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)
        dataset.append(data)

    filenames = [fn.split(".")[0] for fn in filenames]
    data = []

    for i in range(len(dataset)):
        task = dataset[i]
        file_name = filenames[i]

        inputs = [grid['input'] for grid in task]
        outputs = [grid['output'] for grid in task]

        data.append({
            'task': file_name,
            'data_num': len(inputs),
            'input': inputs,
            'output': outputs,
        })

    df = pd.DataFrame(data)
    return df


def change_task_property(property_file, dataset):
    with open(property_file, 'r', encoding='utf-8') as f:
        property_json = json.load(f)
    for data in dataset:
        task_id = data['task']
        if any(key in property_json[task_id] for key in TaskPropertyList):
            continue

        print(f"Task ID: {task_id} press ENTER to start, 'skip' to skip")
        user_input = input()
        if user_input.lower() == 'skip':
            continue
        elif user_input != '':
            return
        for i in range(data['data_num']):
            input_data = data['input'][i]
            output_data = data['output'][i]
            print("Input: Original / Rotated / Transposed / Permutated")
            # render_grid(input_data)
            render_grids_parallel([
                input_data,
                transform_array(input_data, ['rt']),
                transform_array(input_data, ['tp']),
                transform_array(input_data, ['perm9182736450']),
            ])
            print("Output: Original / Rotated / Transposed / Permutated")
            # render_grid(output_data)
            render_grids_parallel([
                output_data,
                transform_array(output_data, ['rt']),
                transform_array(output_data, ['tp']),
                transform_array(output_data, ['perm9182736450']),
            ])

            # Display possible actions
            print("press ENTER to check other example, Any other keys to stop and fill task property.")
            user_input = input()
            if user_input != '':
                break
            sys.stdout.write('\033[F')  # Move cursor up one line
            sys.stdout.write('\033[K')  # Clear the line
            sys.stdout.write('\033[F')  # Move cursor up one line
            sys.stdout.write('\033[K')  # Clear the line

        for property in TaskPropertyList:
            while True:
                user_input = input(f"Can {property}? (Y/ N / skip): ")
                if user_input in ["Y", "y"]:
                    property_json[task_id][property] = True
                    break
                elif user_input in ["N", "n"]:
                    property_json[task_id][property] = False
                    break
                elif user_input.lower() == 'skip':
                    break
                else:
                    print("Illegal response")

        with open(property_file, 'w', encoding='utf-8') as f:
            json.dump(property_json, f, indent=4, ensure_ascii=False)


def transform_array(array, transforms, apply_perm=True):
    if array is None:
        return None
    array = np.asarray(array)
    for tf in transforms:
        if tf == 'tp':
            array = np.swapaxes(array, 0, 1)
        if tf == 'rt':
            array = np.rot90(array)
        if apply_perm and tf.startswith('perm'):
            array = permute_array(array, tf)
    return array


def permute_array(array, descriptor):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    if len(permutation) == 0:
        permutation = random.sample(range(10), 10)
    assert sorted(permutation) == list(range(10))
    array = np.asarray(array)
    assert array.ndim == 2
    array = np.asarray(permutation)[array]
    return array


data_path = "/home/student/joonseo_workspace/2025DL-team-project/workspace/dataset"
df = load_data(data_path)

dataset = Dataset.from_pandas(df)
property_file = 'task_property.json'
change_task_property(property_file, dataset)
