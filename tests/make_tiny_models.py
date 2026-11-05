"""Create tiny random GPT-2 and RoBERTa models (+ tokenizers trained on fixture text) so the
whole pipeline can run offline on CPU in a minute.

    python tests/make_tiny_models.py [out_dir]
"""
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, processors
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, RobertaConfig,
                          RobertaForSequenceClassification)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emodial.emotions import EMOTIONS, LABEL2ID  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".smoke/tiny")
FIX = Path(__file__).parent / "fixtures" / "raw"
corpus = [p.read_text(errors="ignore") for p in FIX.rglob("*") if p.is_file()]


def bpe(specials):
    t = ByteLevelBPETokenizer()
    t.train_from_iterator(corpus, vocab_size=600, min_frequency=1, special_tokens=specials)
    return t


# GPT-2 style generator
g = bpe(["<|endoftext|>"])
gtok = PreTrainedTokenizerFast(tokenizer_object=g._tokenizer, bos_token="<|endoftext|>",
                               eos_token="<|endoftext|>", unk_token="<|endoftext|>")
gtok.save_pretrained(OUT / "gpt2")
GPT2LMHeadModel(GPT2Config(vocab_size=len(gtok), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                           bos_token_id=gtok.eos_token_id, eos_token_id=gtok.eos_token_id)).save_pretrained(OUT / "gpt2")

# RoBERTa style classifier (recogniser in smoke runs, and stand-in judge)
r = bpe(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
r._tokenizer.post_processor = processors.RobertaProcessing(("</s>", r.token_to_id("</s>")), ("<s>", r.token_to_id("<s>")))
rtok = PreTrainedTokenizerFast(tokenizer_object=r._tokenizer, bos_token="<s>", eos_token="</s>", sep_token="</s>",
                               cls_token="<s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
rtok.save_pretrained(OUT / "roberta")
RobertaForSequenceClassification(RobertaConfig(
    vocab_size=len(rtok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
    max_position_embeddings=260, pad_token_id=rtok.pad_token_id, num_labels=len(EMOTIONS),
    id2label=dict(enumerate(EMOTIONS)), label2id=LABEL2ID)).save_pretrained(OUT / "roberta")
print("tiny models written to", OUT)
