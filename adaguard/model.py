"""Offline Transformers runtime shared by training and inference."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('DO_NOT_TRACK', '1')
import math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from .data import messages, answer
from .formatting import parse_guard_response


def load_model(path, device='cpu', dtype='float32'):
    if not Path(path).is_dir():
        raise ValueError('model must be an existing local directory')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer.chat_template = Path(__file__).with_name('chat_template.jinja').read_text().rstrip('\n')
    if not tokenizer.is_fast:
        raise ValueError('a fast tokenizer is required for exact region alignment')
    for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
        if token not in tokenizer.get_vocab():
            raise ValueError('expected a Qwen-compatible ChatML tokenizer')
    tokenizer.eos_token = '<|im_end|>'
    tokenizer.pad_token = '<|endoftext|>'
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=getattr(torch, dtype), attn_implementation='eager').to(device)
    model.config.use_cache = False
    model.eval()  # Keep dropout disabled, including during gradient-based updates.
    return model, tokenizer


def prompt_text(tokenizer, row):
    return tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)


def prompt_ids(tokenizer, row):
    return tokenizer.encode(prompt_text(tokenizer, row), add_special_tokens=False)


def check_length(model, length, limit):
    context = getattr(model.config, 'max_position_embeddings', limit)
    if length > min(context, limit):
        raise ValueError('input exceeds context budget; no silent truncation is performed')


def sft_example(tokenizer, row):
    prompt = prompt_text(tokenizer, row)
    target = answer(row)
    text = prompt + target + tokenizer.eos_token
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = len(prompt)
    label_start = start + target.index('<label>')
    label_end = start + len(target)
    weights = []
    for left, right in enc['offset_mapping']:
        if left < start < right:
            raise ValueError('token crosses prompt/answer boundary')
        weights.append(0. if right <= start else 4. if left < label_end and right > label_start else 1.)
    return enc['input_ids'], weights


def regions(tokenizer, response_ids):
    # EOS belongs to structure/verdict; never infer boundaries from character counts alone.
    eos = tokenizer.eos_token_id
    ended = bool(response_ids and response_ids[-1] == eos)
    body = response_ids[:-1] if ended else response_ids
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    parsed = parse_guard_response(text)
    mask = [False] * len(response_ids)
    if not parsed.valid:
        return text, mask, False, ended
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc['input_ids']) != list(body):
        return text, mask, False, ended
    begin = len('<analysis>\n')
    end = text.index('\n</analysis>')
    for i, (left, right) in enumerate(enc['offset_mapping']):
        mask[i] = begin <= left < right <= end
    return text, mask, any(mask) and not all(mask), ended


@torch.no_grad()
def generate(model, tokenizer, row, max_new_tokens=640, sample=False, max_length=16000):
    ids = prompt_ids(tokenizer, row)
    check_length(model, len(ids)+max_new_tokens, max_length)
    input_ids = torch.tensor([ids], device=model.device)
    sampling = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0} if sample else {}
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=sample,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True, **sampling)
    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=config)
    response = output[0, len(ids):].tolist()
    return ids, response, regions(tokenizer, response)


def logprobs(model, ids, prompt_length):
    inputs = torch.tensor([ids], device=model.device)
    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
    # Logit at prefix position t-1 predicts response token t.
    return logits[0, prompt_length-1:-1].float().log_softmax(-1).gather(
        -1, inputs[0, prompt_length:, None]).squeeze(-1)


class PrefixValue(torch.nn.Module):
    def __init__(self, causal_model):
        super().__init__()
        self.backbone = causal_model.base_model
        self.head = torch.nn.Linear(causal_model.config.hidden_size, 1, bias=False,
                                    device=causal_model.device, dtype=next(causal_model.parameters()).dtype)
        torch.nn.init.zeros_(self.head.weight)

    def forward(self, ids, prompt_length):
        device = next(self.parameters()).device
        inputs = torch.tensor([ids], device=device)
        hidden = self.backbone(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).last_hidden_state
        raw = self.head(hidden[0, prompt_length-1:-1]).squeeze(-1).float()
        return (2 / math.pi) * torch.atan(raw)


def save_actor(model, tokenizer, output):
    model.config._name_or_path = ''
    tokenizer.init_kwargs.pop('name_or_path', None)
    tokenizer.name_or_path = ''
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
