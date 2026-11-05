from pathlib import Path

from emodial.data.common import iter_examples, merge_consecutive
from emodial.data.erc import parse_dailydialog_dir, parse_meld_csv
from emodial.data.generation import parse_ed_csv, parse_esconv
from emodial.generator.data import attach_emotions

RAW = Path(__file__).parent / "fixtures" / "raw"


def test_ed_parsing_roles_and_commas():
    d = parse_ed_csv(RAW / "empatheticdialogues" / "train.csv", "train")
    assert d and d[0]["turns"][0]["role"] == "user"
    assert all("_comma_" not in t["text"] for x in d for t in x["turns"])


def test_esconv_split_is_complete_and_disjoint():
    s = parse_esconv(RAW / "ESConv.json")
    ids = [x["id"] for v in s.values() for x in v]
    assert len(ids) == len(set(ids)) == 30


def test_erc_parsers():
    assert parse_dailydialog_dir(RAW / "dailydialog", "train")
    assert parse_meld_csv(RAW / "meld" / "train_sent_emo.csv", "train")


def test_merge_consecutive():
    t = merge_consecutive([{"role": "user", "text": "a"}, {"role": "user", "text": "b"}, {"role": "listener", "text": "c"}])
    assert t == [{"role": "user", "text": "a b"}, {"role": "listener", "text": "c"}]


def test_examples_and_delta():
    dlg = [{"id": "x", "source": "ed", "turns": [
        {"role": "user", "text": "u0"}, {"role": "listener", "text": "l1"},
        {"role": "user", "text": "u2"}, {"role": "listener", "text": "l3"}]}]
    ex = list(iter_examples(dlg, 3))
    assert [e["response"] for e in ex] == ["l1", "l3"]
    assert ex[0]["prev_key"] is None and ex[1]["prev_key"] == "x#0" and ex[1]["cur_key"] == "x#2"
    cache = {"x#0": [1, 0, 0, 0, 0, 0, 0], "x#2": [0, 0, 0, 1, 0, 0, 0]}
    rows = attach_emotions(ex, cache)
    assert rows[0]["shift"] == 0 and rows[1]["shift"] == 2
