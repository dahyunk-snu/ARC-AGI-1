import unsloth
import os


from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train
from unsloth import UnslothTrainingArguments as TrainingArguments
from peft import PeftModel

import torch
import torch.nn.functional as F
from typing import List
import numpy as np
import random
import gc
from .utils import InputMaskingDataCollator
from .utils import load_unsloth_4bit, save_model_and_tokenizer, keep_single_char_tokens
from tqdm.auto import tqdm
from datasets import Dataset
from .task_clusterization.clusterization import classify_cluster
from transformers import GenerationConfig

from .arc_loader import ArcDataset
from .inference_tools import inference_run, infer_task
from .selection import EvalTool

class ARCSolver:
    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """

        config_path = "artifacts/config/config.yml"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        
        version = "-05-24-MoE"
        self.adapter_path_common = "artifacts/checkpoint-final-test" + version
        self.training_output_dir_common = "artifacts/training_checkpoints" + version

        origin_version = "-05-21-LowRank"
        self.adapter_path_origin = "artifacts/checkpoint-final-test" + origin_version
        
        # (1) 모델 불러오기
        print("\n>>> Load model...\n")
        self.model, self.tokenizer = load_unsloth_4bit(model_id)
        print("\n>>> Successfully loaded model\n")

        # (1-1) 기본 모델이면 임베딩 차원을 축소
        print("\n>>> Shrink Embedding dimensions...\n")
        keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+self.tokenizer.tokenize('\n')
        keep_single_char_tokens(self.model, self.tokenizer, keep=keep_tok, remove_unk=True)
        print("\n>>> Successfully shrinked embedding dimensions\n")
        
        # (2) Prompt 양식 설정 및 인코딩
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + self.tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=128000,
        )

        self.preprompt = self.tokenizer.encode(self.fmt_opts['preprompt'], add_special_tokens=False)
        self.query_beg = self.tokenizer.encode(self.fmt_opts['query_beg'], add_special_tokens=False)
        self.reply_beg = self.tokenizer.encode(self.fmt_opts['reply_beg'], add_special_tokens=False)
        self.reply_end = self.tokenizer.encode(self.fmt_opts['reply_end'], add_special_tokens=False)

        self.pixel_ids = [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
        self.lines_sep = self.tokenizer.encode(self.fmt_opts['lines_sep'], add_special_tokens=False)[0]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.prompt_max_len = 256

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format
        Args:
            ids (List[int]): LLM generated token list
        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}

        for idx in ids:
            if idx == self.lines_sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens
        Args:
            grid (List[List[int]]): 2D grid
        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.lines_sep)
        return ids

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): contains training data, test input
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        test_data = datapoint['test'][0]

        prompt = self.preprompt.copy()
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            prompt += self.query_beg + inp + self.reply_beg
            prompt += out + self.reply_end

        inp = test_data['input']
        out = test_data['output']
        inp = self.format_grid(inp)

        prompt += self.query_beg + inp + self.reply_beg

        if out is not None:
            out = self.format_grid(out)
            prompt += out + self.reply_end

        return prompt

    def train(self, train_dataset_splitted, pretrained=False, classes=None):
        num_experts = len(train_dataset_splitted)
        if classes is None:
            classes = range(num_experts)
        self.adapter_path_list = []
        self.training_output_dir_list = []

        for i in range(num_experts):
            self.adapter_path_list.append(self.adapter_path_common + f"/{i}")
            self.training_output_dir_list.append(self.training_output_dir_common + f"/{i}")

        for i in classes:
        # for i in range(3):
        # for i in range(num_experts):
            if i == 5 or i == 3:
                epoch = 2
            else:
                epoch = 1
            train_dataset = train_dataset_splitted[i]
            self.adapter_path = self.adapter_path_list[i]
            self.training_output_dir = self.training_output_dir_list[i]
            print(f"\n>>> Expert Fin-tuning of {i}/{num_experts-1}.\n")        
            # 0) Load pretrained model
            if pretrained:
                print("\n>>> Load a pretrained model...\n")    
                # self.model_MoE = PeftModel.from_pretrained(self.model, self.adapter_path_origin, is_trainable=True).half()
                lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
                self.model_MoE = FastLanguageModel.get_peft_model(
                    model=self.model,
                    target_modules=lora_layers,
                    r=64,#256
                    lora_alpha=32,#32#24
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,
                    loftq_config=None,
                )
                self.model_MoE.load_adapter(
                    self.adapter_path_origin,   # LoRA weight 디렉터리
                    adapter_name="default",     # 덮어쓸 이름
                    is_trainable=True,          # 이어 학습 목적
                    replace_weights=True,       # 기존 weight 덮어쓰기
                )
                print("\n>>> Successfully loaded a pretrained model\n")

            else:
                # 1) Prepare PEFT/LoRA
                print("\n>>> Preparing PEFT/LoRA...\n")
                lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
                self.model = FastLanguageModel.get_peft_model(
                    model=self.model,
                    target_modules=lora_layers,
                    r=64,#256
                    lora_alpha=32,#24
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,
                    loftq_config=None,
                )
            print("\n>>> Successfully preparing PEFT/LoRA\n")


            # 2) Preprocess Dataset
            examples = []
            for entry in tqdm(train_dataset, desc="Prepare training dataset..."):
                input_ids = self.format_prompt(entry)
                examples.append({"input_ids": input_ids})
                # examples.append({"input_ids": input_ids[-self.prompt_max_len:]})
            dataset = Dataset.from_list(examples)
            print("\n>>> Successfully prepared training dataset\n")

            # 3) Data collator (padding & mask)
            data_collator = InputMaskingDataCollator(
                instruction_template=self.fmt_opts['query_beg'],
                response_template=self.fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=self.tokenizer,
                mask_first_n_examples=1,
            )

            # 4) Training
            os.makedirs(self.training_output_dir, exist_ok=True)

            FastLanguageModel.for_training(self.model_MoE)
            self.model_MoE.print_trainable_parameters()
            self.tokenizer.padding_side = 'right'
            training_args = TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                # warmup_ratio=0.25,
                warmup_ratio=0.1,
                num_train_epochs=epoch,
                learning_rate=1e-4,
                embedding_learning_rate=1e-5,
                fp16=True,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='linear',
                # lr_scheduler_type='cosine',
                seed=42,

                output_dir=self.training_output_dir,
                save_strategy="steps",
                save_steps=500,
                save_total_limit=5,
                report_to='none',
            )

            #run training
            trainer = Trainer(
                model=self.model_MoE,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
                packing=False,
                data_collator=data_collator,
                max_seq_length=self.fmt_opts['max_tokens'],
            )

            print("\n>>> Starting fine-tuning...\n")
            trainer_stats = unsloth_train(trainer)

            # 5) Save adapter
            print(f"\n>>> Training complete. Saving adapter to {self.adapter_path}")
            save_model_and_tokenizer(self.adapter_path, self.model_MoE, self.tokenizer)
            print(f"\n>>> Fin-tuning of {i}/{num_experts} finished and adapter saved.\n")        

            # # 6) detatch adapter
            # self.model, _ = self.model_MoE.unload()   # ← LoRA·PEFT 래핑 전부 제거
            # torch.cuda.empty_cache()                  # 캐시 비우기
            # gc.collect()                              # Python GC

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """        
        model_dir = "artifacts/cluster_model_fixed"
        label,s = classify_cluster(model_dir, examples, print=False)
        if s < 0.5:
            self.model.set_adapter(
                "default"
            )
        else:
            self.model.set_adapter(
                f"expert_{label}"
            )
        eval_dataset = [{
            'task': 'task',
            'train': examples,
            'test_input': [{'input': questions_input}],
            'test_output': None,
            }]
        arc_eval_set = ArcDataset.load_from_dataset(eval_dataset)

        infer_aug_opts = dict(tp='all', rt='all', perm=False, shfl_ex=False, seed=10000)
        infer_dataset = arc_eval_set.augment(**infer_aug_opts)
        eval_tool = EvalTool(n_guesses=1)

        max_new_tokens = infer_dataset.max_new_tokens(**self.fmt_opts)
        if 'max_tokens' in self.fmt_opts:
            fmt_opts = {**self.fmt_opts, 'max_tokens': self.fmt_opts['max_tokens'] - max_new_tokens, 'len_name': 'input'}
        
        _, tasks =next(iter(infer_dataset.grouped_keys().items()))
        task = tasks[0]
        res = infer_task(
            keys=task, 
            dataset=infer_dataset, 
            fmt_opts=fmt_opts, 
            max_new_tokens=max_new_tokens,
            model_tok=(self.model, self.tokenizer),
            min_prob=0.1,
            aug_score_opts=infer_aug_opts,
        )
        if res:
            output = eval_tool.get_best(res)
            return output

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        train_input = np.array(examples[0]['input'])
        train_output = np.array(examples[0]['output'])
        test_input = np.array(questions_input)

        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
            grid = grid[:x, :y]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))
        return np.random.randint(0, 10, (x, y))
        
        
    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        num_experts = 6
        self.adapter_path_list = []

        for i in range(num_experts):
            self.adapter_path_list.append(self.adapter_path_common + f"/{i}")
          
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path_origin, adapter_name = "default",is_trainable=False)

        for i in range(num_experts):
            self.model.load_adapter(
                self.adapter_path_list[i],   # LoRA weight 디렉터리
                adapter_name=f"expert_{i}",  # 덮어쓸 이름
                is_trainable=False,           # 이어 학습 목적 아님
            )

        FastLanguageModel.for_inference(self.model)
        self.model.to(self.device)
        self.model.eval()

if __name__ == "__main__":
    solver = ARCSolver()