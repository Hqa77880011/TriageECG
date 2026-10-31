"""Synthetic data/gradient/policy check, including known-only accepted risk."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .__main__ import Triage, calibrate, evaluate, plot, prepare, run, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/validation")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    c = json.loads(Path("config.json").read_text())
    c.update(classes=2, width=8, epochs=1, batch_size=8, threshold_points=4,
             coverage=.8, coverage_tolerance=.21, minimum_retention=.5,retention_min_support=1)
    rng = np.random.default_rng(c["seed"])
    roles = np.repeat(["train", "dev", "cal", "test"], [16, 16, 16, 16])
    labels = np.tile([0, 1, 0, 1, -1, -1, 0, 1], 8)
    labels[:16] = np.arange(16) % 2
    t = np.linspace(0, 2 * np.pi, 64)
    x = np.sin((labels.clip(0) + 1)[:, None, None] * t[None, None])
    x = (x + rng.normal(0, .2, x.shape)).astype(np.float32)
    source = out / "input.npz"
    np.savez(source, x=x, y=labels, patient=np.arange(64).astype(str), split=roles,
             defect=(np.arange(64) % 4 == 0).astype(int), readable=np.ones(64, bool))
    prepare(c, out, source)
    train(c, out)
    model = Triage(1, c)
    batch = torch.tensor(x[:8])
    logits, z, defect = model(batch)
    loss = F.cross_entropy(logits, torch.tensor(labels[:8]))
    loss += F.binary_cross_entropy_with_logits(defect, torch.zeros(8)) + z.square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.technical[-1].weight.grad.abs().sum() > 0
    before = model.classifier.weight.detach().clone()
    torch.optim.Adam(model.parameters()).step()
    assert not torch.equal(before, model.classifier.weight)
    model.eval()
    torch.save(model.state_dict(), out / "roundtrip.pt")
    restored = Triage(1, c).eval()
    restored.load_state_dict(torch.load(out / "roundtrip.pt", weights_only=True))
    with torch.no_grad():
        delta = max(float((a - b).abs().max()) for a, b in zip(model(batch), restored(batch)))
    assert delta <= 1e-7
    calibrate(c, out)
    assert json.loads((out / "policy.json").read_text())["feasible"]
    run(c, out)
    evaluate(c, out)
    plot(out)
    # Two accepted known records are correct. Accepting an unknown is a separate endpoint.
    hand = out / "hand_metrics"
    hand.mkdir(exist_ok=True)
    np.savez(hand / "predictions.npz", probability=np.array([[.9,.1],[.1,.9],[.8,.2],[.6,.4]]),
             y=np.array([0,1,-1,-1]), action=np.array([0,0,0,2]),
             readable=np.ones(4,bool), technical=np.zeros(4),
             novelty=np.array([.1,.2,.8,.9]))
    evaluate(c, hand)
    metrics = pd.read_csv(hand / "metrics.csv").iloc[0]
    assert metrics.accepted_error == 0
    assert metrics.accepted_unknown_rate == .5
    assert metrics.semantic_auroc == 1
    assert metrics.minimum_class_retention == 1
    assert metrics.wrong_action_rate == .25
    report = dict(synthetic_function_check=True, seed=c["seed"],
                  loss=float(loss.detach()), roundtrip_max_abs=delta,
                  hand_accepted_known_error=float(metrics.accepted_error))
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
