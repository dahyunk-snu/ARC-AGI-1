import json
from tokenizers import Tokenizer
import torch
from trl import DataCollatorForCompletionOnlyLM
import os

class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch['labels'])):
            for _ in range(self.mask_first_n_examples):
                label = batch['labels'][i]

                # (1) 전체 unmasked 위치 찾기
                unmasked = (label != -100).nonzero(as_tuple=False)
                if unmasked.numel() == 0:
                    break
                beg_pos = unmasked.min().item()

                # (2) beg_pos 이후 첫 번째 masked 위치 찾기
                next_mask = (label[beg_pos:] == -100).nonzero(as_tuple=False)
                if next_mask.numel() == 0:
                    break  # 더 이상 마스킹 구간이 없음
                mid_pos = next_mask.min().item() + beg_pos

                # (3) 마지막 unmasked 위치
                end_pos = unmasked.max().item() + 1

                if mid_pos < end_pos:
                    label[beg_pos:mid_pos] = -100
        return batch

def load_unsloth_4bit(model_path):
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )

def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path, save_embedding_layers=True)
    tokenizer.save_pretrained(store_path)
    to_delete = os.path.join(store_path, 'tokenizer.model')  # delete file, as it interferes with token removal
    if os.path.isfile(to_delete):
        os.remove(to_delete)

def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))

    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    
    return tokens

def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    """
    @load
    {
        "model": { "type": "BPE", "vocab": { "a":0, "b":1, … }, "merges": [ … ] },
        "added_tokens": [ … ],
        "post_processor": { … }
    }
    """
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    # 1) Get special tokens to keep
    if keep_special:
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    # 2) Remove unknown token
    if remove_unk: 
        keep_indices -= {tokenizer.unk_token_id}

    # 3) Build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # 4) Update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))

    if remove_unk:
        tokenizer.unk_token = None

    return mapping

def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # 1) Copy embeddings to keep
        # Sorted old_id w/ new_id
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(
            model.get_input_embeddings().weight.data,   # (old_vocab_size, embedding_dim)
            0,                                          # based on row=0
            row_select                                  # list of old_ids
            )
        # row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(
            model.get_output_embeddings().weight.data,  # (old_vocab_size, embedding_dim)
            0,                                          # based on row=0
            row_select                                  # list of old_ids
            )

        # 2) Resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # 3) Set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # 4) Map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def keep_single_char_tokens(model, tokenizer, keep=None, **kwargs):
    # 1) Keep tokens that were passed
    keep_indices = set(tokenizer.vocab[t] for t in keep)

    # 2) Keep tokens used by model
    for config in [model.config, model.generation_config]:
        for k, v in config.to_dict().items():
            if k.endswith('token_id'):
                keep_indices.update(v if isinstance(v, list) else [v])

    # 3) Shrink tokenizer vocab
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping
