"""Offline integration tests with a randomly initialized miniature Qwen model."""
import json
import sys
from pathlib import Path
import pytest
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from adaguard.data import load, messages, answer
from adaguard import model

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    torch.set_num_threads(1)
    path=tmp_path_factory.mktemp('local-model')
    rows=load(ROOT/'data/examples.jsonl')
    tok=Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=1000,special_tokens=['[UNK]','<|im_start|>','<|im_end|>','<|endoftext|>'],
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([json.dumps(messages(r))+answer(r) for r in rows],trainer)
    fast=PreTrainedTokenizerFast(tokenizer_object=tok,eos_token='<|im_end|>',pad_token='<|endoftext|>',unk_token='[UNK]')
    fast.save_pretrained(path)
    config=Qwen3Config(vocab_size=len(fast),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096,
        tie_word_embeddings=True,eos_token_id=fast.eos_token_id,pad_token_id=fast.pad_token_id)
    torch.manual_seed(7)
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path,rows


def test_token_regions_and_offline_inference(tiny):
    from adaguard.infer import Guard
    path,rows=tiny
    guard=Guard(path)
    tokenizer=guard.tokenizer
    row=rows[0]
    ids,weights=model.sft_example(tokenizer,row)
    assert 0 in weights and 1 in weights and 4 in weights
    response=tokenizer.encode(answer(row),add_special_tokens=False)+[tokenizer.eos_token_id]
    text,mask,structured,ended=model.regions(tokenizer,response)
    assert text==answer(row) and structured and ended and any(mask) and not mask[-1]
    result=guard.predict(row['policy'],row['content'],max_new_tokens=8,max_length=4096)
    assert result['status'] in {'OK','INVALID'}
    if result['status']=='INVALID':
        assert result['unsafe'] is None and result['violated_ids'] is None
    with pytest.raises(ValueError,match='context budget'):
        guard.predict(row['policy'],row['content'],max_new_tokens=8,max_length=10)


def test_sft_checkpoint(tiny,tmp_path,monkeypatch):
    from adaguard import train_sft
    path,_=tiny
    out=tmp_path/'sft'
    monkeypatch.setattr(sys,'argv',['train_sft','--model',str(path),'--train',str(ROOT/'data/examples.jsonl'),
                                   '--output',str(out),'--max-length','4096'])
    train_sft.main()
    trained,tokenizer=model.load_model(out/'epoch-1')
    assert len((out/'metrics.jsonl').read_text().splitlines())==4
    assert tokenizer.chat_template


def test_safepo_nonzero_update(tiny,tmp_path,monkeypatch):
    from adaguard import train_safepo
    path,_=tiny
    initial,_=model.load_model(path)
    before={k:v.detach().clone() for k,v in initial.state_dict().items()}
    counter=0
    def controlled_generate(actor,tokenizer,row,**kwargs):
        nonlocal counter
        counter+=1
        text=answer(row) if counter%2 else '<analysis>\nNo violation.\n</analysis>\n<label>unknown</label>'
        prompt=model.prompt_ids(tokenizer,row)
        response=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
        return prompt,response,model.regions(tokenizer,response)
    monkeypatch.setattr(model,'generate',controlled_generate)
    out=tmp_path/'safepo'
    monkeypatch.setattr(sys,'argv',['train_safepo','--model',str(path),'--train',str(ROOT/'data/examples.jsonl'),
                                   '--output',str(out),'--steps','1','--max-length','4096'])
    train_safepo.main()
    trained,_=model.load_model(out/'step-1')
    assert counter==8
    assert any(not torch.equal(before[k],v) for k,v in trained.state_dict().items())
    record=json.loads((out/'metrics.jsonl').read_text())
    assert record['sequence_adv_abs']>0
    assert (out/'step-1/quality.json').exists()
