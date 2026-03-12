import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 실제 Transformers 기반 로직 담당
class HFBackend:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
                             model_name,
                             torch_dtype=torch.float16,
                             device_map="auto"
                         )
        self.device    = device

    def generate(self, prompt: str, max_tokens: int, temperature: float, n: int):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n
        )
        # 디코딩
        return [ self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs ]

    def embed(self, texts: list[str]):
        # 간단 예시: 마지막 hidden state 평균 사용
        embs = []
        for t in texts:
            inputs = self.tokenizer(t, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
            embs.append(hidden.mean(dim=1).squeeze().cpu().tolist())
        return embs

# 2. OpenAI SDK 호환 인터페이스를 흉내 내는 Shim
class ShimOpenAI:
    def __init__(self, hf_kwargs):
        self._hf = HFBackend(**hf_kwargs)
        # .chat.completions.create 인터페이스 흉내
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=self._chat_create
            )
        )
        # .embeddings.create 인터페이스 흉내
        self.embeddings = types.SimpleNamespace(
            create=self._embed_create
        )

    def _chat_create(self, *, model, messages, temperature, max_tokens, top_p, n):
        # messages를 하나의 문자열로 직렬화
        prompt = ""
        for m in messages:
            role = m["role"].upper()
            prompt += f"[{role}]: {m['content']}\n"
        # 백엔드에 위임
        raws = self._hf.generate(prompt, max_tokens, temperature, n)
        # OpenAI SDK 형태로 포장
        choices = [{"message":{"role":"assistant","content":r}} for r in raws]
        usage = types.SimpleNamespace(prompt_tokens=0, total_tokens=0)
        return types.SimpleNamespace(choices=choices, usage=usage)

    def _embed_create(self, *, model, input, encoding_format=None):
        texts = input if isinstance(input, list) else [input]
        raws = self._hf.embed(texts)
        data = [ types.SimpleNamespace(embedding=e) for e in raws ]
        usage = types.SimpleNamespace(prompt_tokens=0, total_tokens=0)
        return types.SimpleNamespace(data=data, usage=usage)
