import torch

from emodial.generator.etg import EmotionPrompt


def _inputs(b=3, k=7):
    p = torch.softmax(torch.randn(b, k), -1)
    return p, p - torch.softmax(torch.randn(b, k), -1)


def test_shapes_and_gates():
    p, d = _inputs()
    for v in ("b1", "etg_nodelta", "etg_nogate", "etg"):
        s, g = EmotionPrompt(v, 7, 4, 16)(p, d)
        assert s.shape == (3, 4, 16)
        if v == "b1":
            assert g is None
        else:
            assert g.shape == (3,) and torch.all((g >= 0) & (g <= 1))


def test_nogate_is_always_one_and_nodelta_ignores_delta():
    p, d = _inputs()
    _, g = EmotionPrompt("etg_nogate", 7, 4, 16)(p, d)
    assert torch.allclose(g, torch.ones_like(g))
    m = EmotionPrompt("etg_nodelta", 7, 4, 16)
    assert torch.allclose(m(p, d)[0], m(p, torch.zeros_like(d))[0])


def test_etg_uses_delta():
    torch.manual_seed(0)
    m = EmotionPrompt("etg", 7, 4, 16, init_std=0.5)
    p, d = _inputs()
    assert not torch.allclose(m(p, d)[0], m(p, torch.zeros_like(d))[0])


def test_parameter_count_matches_proposal():
    # DialoGPT-small: hidden 768, m = 4 -> about 70.7k parameters for full ETG
    assert EmotionPrompt("etg", 7, 4, 768).n_params() == 70665
