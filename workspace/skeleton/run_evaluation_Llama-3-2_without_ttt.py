# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from unsloth import FastLanguageModel
from arc.utils import load_unsloth_4bit, keep_single_char_tokens

from diskcache import Cache
from arc.arc_loader import ArcDataset
from arc.inference_tools import inference_run
from arc.selection import EvalTool

import torch
from evaluate import load_data
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load model
token = os.environ.get("HF_TOKEN", None)
base_model = "meta-llama/Llama-3.2-3B-Instruct"
# 현재는 merge까지 진행한 후 eval.
adapter_path = "artifacts/checkpoint-final-test-05-13/"

model, tokenizer = load_unsloth_4bit(base_model)
keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=2048,
)



# input paths
data_path = "/workspace/dataset"
N_data = 10
df = load_data(data_path)
eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(N_data))

# output paths
output_path = 'output_evaluation_Llama-rearc_without_ttt'
inference_cache = os.path.join(output_path, 'inference_cache')
submission_file = os.path.join(output_path, 'submission.json')

# load evaluation dataset
arc_eval_set = ArcDataset.load_from_dataset(eval_dataset)

# preparation for model.
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

FastLanguageModel.for_inference(model)
model.to(device)
model.eval()

# evaluation
infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
infer_dataset = arc_eval_set.augment(**infer_aug_opts)
model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
eval_tool = EvalTool(n_guesses=2)
inference_results = inference_run(
    model_tok=(model, tokenizer),
    fmt_opts=fmt_opts,
    dataset=infer_dataset,
    min_prob=0.1,
    aug_score_opts=infer_aug_opts,
    callback=eval_tool.process_result,
    cache=model_cache,
)

# write submission
with open(submission_file, 'w') as f:
    json.dump(arc_eval_set.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))