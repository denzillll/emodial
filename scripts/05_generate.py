"""Step 5 - generate replies on the fixed test subset (identical for every system).

    python scripts/05_generate.py --variant etg --seed 42 [--swap]

--swap also generates with p_t / Delta_t taken from other test contexts (the "is the
emotion prompt actually used?" check); only meaningful for emotion variants.
"""
import _bootstrap  # noqa: F401
import random

from emodial.config import cli, device, seed_everything
from emodial.data.common import write_jsonl
from emodial.generator.etg import VARIANTS
from emodial.generator.model import EmoDial
from emodial.pipeline import Paths, load_emotion_cache, test_subset


def extra(p):
    p.add_argument("--variant", choices=VARIANTS, default="etg")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--swap", action="store_true")


args, cfg = cli(__doc__, extra)
P, dev = Paths(cfg), device()
seed = args.seed if args.seed is not None else cfg["seed"]
seed_everything(seed)
model = EmoDial.load(P.model_dir(args.variant, seed)).to(dev)
rows = test_subset(cfg, P, load_emotion_cache(P))
gen_cfg = {k: v for k, v in cfg["generation"].items() if k != "batch_size"}
bs, maxc = cfg["generation"]["batch_size"], cfg["generator"]["max_context_tokens"]


def run(rs):
    out = []
    for i in range(0, len(rs), bs):
        texts, gates = model.generate(rs[i:i + bs], gen_cfg, maxc, dev)
        for r, t, g in zip(rs[i:i + bs], texts, gates):
            out.append({**r, "generated": t, "gate": g})
        print(f"  {min(i + bs, len(rs))}/{len(rs)}", end="\r")
    print()
    return out


tag = f"{args.variant}_s{seed}"
n = write_jsonl(run(rows), P.replies / f"{tag}.jsonl")
print(f"wrote {n} replies -> {P.replies / f'{tag}.jsonl'}")

if args.swap and args.variant != "b0":
    sub = rows[: cfg["eval"]["swap_examples"]]
    donors = sub[:]
    random.Random(seed).shuffle(donors)
    swapped = [{**r, "p": d["p"], "delta": d["delta"], "shift": d["shift"], "swapped_from": d["id"]}
               for r, d in zip(sub, donors)]
    n = write_jsonl(run(swapped), P.replies / f"{tag}_swap.jsonl")
    print(f"wrote {n} swapped-emotion replies")
