"""Step 4 - train the generator.

Stage 1 (once):   python scripts/04_train_generator.py --stage base
Stage 2 (each):   python scripts/04_train_generator.py --variant etg --seed 42
                  variants: b0 b1 etg_nodelta etg_nogate etg

Every stage-2 system starts from the same stage-1 checkpoint and trains for the same number
of steps, so differences between systems come only from the emotion input.
"""
import _bootstrap  # noqa: F401
import json

from emodial.config import cli, device, seed_everything
from emodial.generator.etg import VARIANTS
from emodial.generator.model import EmoDial
from emodial.generator.train import train
from emodial.pipeline import Paths, examples, load_emotion_cache


def extra(p):
    p.add_argument("--stage", choices=["base", "variant"], default="variant")
    p.add_argument("--variant", choices=VARIANTS, default="etg")
    p.add_argument("--seed", type=int, default=None)


args, cfg = cli(__doc__, extra)
P, gc, dev = Paths(cfg), cfg["generator"], device()
seed = args.seed if args.seed is not None else cfg["seed"]
seed_everything(seed)
cache = load_emotion_cache(P)
train_rows = examples(cfg, P, "train", cache, cfg["data"]["max_train_examples"], seed=cfg["seed"])
valid_rows = examples(cfg, P, "valid", cache, cfg["data"]["max_valid_examples"], seed=cfg["seed"])
print(f"train {len(train_rows)}  valid {len(valid_rows)}  on {dev}")

if args.stage == "base":
    model = EmoDial.from_lm(gc["model_name"], "b0", gc["m"])
    out = P.generator / "base"
    summary = train(model, train_rows, valid_rows, gc, gc["base"], out, dev)
else:
    base = P.generator / "base" / "lm"
    if not base.exists():
        raise SystemExit("Run --stage base first.")
    model = EmoDial.from_lm(str(base), args.variant, gc["m"])
    if model.prompt is not None:
        print(f"{args.variant}: emotion prompt has {model.prompt.n_params():,} parameters")
    out = P.model_dir(args.variant, seed)
    summary = train(model, train_rows, valid_rows, gc, gc["variant"], out, dev)
(out / "train_summary.json").write_text(json.dumps(summary, indent=2))
print(f"saved to {out}  best valid ppl {summary['best_valid_ppl']:.3f}")
