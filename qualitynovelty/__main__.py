import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 7, padding=3), nn.ReLU(), nn.Conv1d(width, width, 7, padding=3))

    def forward(self, x):
        return F.relu(x + self.net(x))


class Triage(nn.Module):
    def __init__(self, channels, c):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(channels, c['width'], 7, padding=3), nn.ReLU(),
                                     Residual(c['width']), nn.AvgPool1d(4), Residual(c['width']), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = nn.Linear(c['width'], c['classes'])
        self.technical = nn.Sequential(nn.Linear(c['width'] + 3 * channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.encoder(x)
        summary = torch.cat([x.std(-1), (x.abs() > 4).float().mean(-1), (x == 0).float().mean(-1)], 1)
        logits = self.classifier(z)
        defect = self.technical(torch.cat([z, summary], 1)).squeeze(-1)
        return logits, z, defect


def prepare(c, out, source):
    d = dict(np.load(source, allow_pickle=False))
    required = {'x', 'y', 'patient', 'split', 'defect', 'readable'}
    if not required <= set(d):
        raise ValueError('Requires x,y,patient,split,defect,readable; unknown y=-1')
    for patient in np.unique(d['patient']):
        if len(np.unique(d['split'][d['patient'] == patient])) != 1:
            raise ValueError('A patient crosses train/dev/cal/test roles')
    train = d['split'] == 'train'
    mean = np.nanmean(d['x'][train], axis=(0, 2), keepdims=True)
    scale = np.nanstd(d['x'][train], axis=(0, 2), keepdims=True).clip(1e-6)
    d['x'] = np.nan_to_num((d['x'] - mean) / scale).astype(np.float32)
    if (d['y'][train] < 0).any():
        raise ValueError('Recognition training must contain known classes only')
    np.savez_compressed(out / 'data.npz', **d)
    np.savez(out / 'normalization.npz', mean=mean, scale=scale)


def train(c, out):
    torch.manual_seed(c['seed'])
    d = np.load(out / 'data.npz')
    x, y = torch.tensor(d['x']), torch.tensor(d['y'], dtype=torch.long)
    model = Triage(x.shape[1], c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    ids = np.flatnonzero(d['split'] == 'train')
    for epoch in range(c['epochs']):
        for selected in torch.tensor(ids)[torch.randperm(len(ids))].split(c['batch_size']):
            current = x[selected]
            logits, z, quality = model(current)
            damaged = current + torch.randn_like(current) * c['corruption_scale']
            dropped = torch.rand(len(current), current.shape[1], 1) < c['lead_dropout']
            damaged = damaged.masked_fill(dropped, 0)
            _, _, corrupt_quality = model(damaged)
            loss = F.cross_entropy(logits, y[selected])
            loss += c['technical_weight'] * (F.binary_cross_entropy_with_logits(quality, torch.tensor(d['defect'][selected], dtype=torch.float32)) +
                                              F.binary_cross_entropy_with_logits(corrupt_quality, torch.ones_like(corrupt_quality)))
            if c['variant'] != 'no-consistency':
                mild = current + torch.randn_like(current) * c['mild_scale']
                loss += c['pair_weight'] * F.mse_loss(z, model(mild)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), out / 'model.pt')


def predict(model, x, c):
    values = []
    model.eval()
    with torch.no_grad():
        for group in torch.tensor(x).split(c['batch_size']):
            logits, z, defect = model(group)
            values.append((logits.softmax(-1).numpy(), z.numpy(), defect.sigmoid().numpy(), (-torch.logsumexp(logits, -1)).numpy()))
    return [np.concatenate([value[k] for value in values]) for k in range(4)]


def novelty_features(z, defect, energy, center, scale, c):
    distances = np.sqrt(((z[:, None] - center[None]) ** 2 / scale[None]).sum(-1))
    parts = [energy[:, None], distances]
    if c['variant'] != 'no-quality':
        parts.append(defect[:, None])
    return np.concatenate(parts, 1)


def route(p, q, u, thresholds, c):
    confidence, quality, novelty = thresholds
    classified = (p.max(1) >= confidence) & (q <= quality) & (u <= novelty)
    actions = np.where(classified, 0, np.where(q > quality, 1, 2))
    if c['variant'] == 'collapsed-actions':
        actions[actions == 1] = 2
    return actions


def calibrate(c, out):
    d = np.load(out / 'data.npz')
    model = Triage(d['x'].shape[1], c)
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True, map_location='cpu'))
    training = d['split'] == 'train'
    _, z, _, _ = predict(model, d['x'][training], c)
    centers = np.stack([z[d['y'][training] == k].mean(0) for k in range(c['classes'])])
    scales = np.stack([z[d['y'][training] == k].var(0).clip(1e-5) for k in range(c['classes'])])
    development = d['split'] == 'dev'
    p, z, q, energy = predict(model, d['x'][development], c)
    feature = novelty_features(z, q, energy, centers, scales, c)
    standardizer = StandardScaler().fit(feature)
    learner = LogisticRegression(max_iter=2000).fit(standardizer.transform(feature), d['y'][development] < 0)
    calibration = d['split'] == 'cal'
    p, z, q, energy = predict(model, d['x'][calibration], c)
    feature = novelty_features(z, q, energy, centers, scales, c)
    u = learner.predict_proba(standardizer.transform(feature))[:, 1]
    labels = d['y'][calibration]
    readable_known = (labels >= 0) & d['readable'][calibration].astype(bool)
    best, selected = np.inf, None
    grid = np.linspace(0, 1, c['threshold_points'])
    for confidence in np.quantile(p.max(1), grid):
        for quality in np.r_[0, np.quantile(q, grid), 1]:
            for novelty in np.r_[0, np.quantile(u, grid), 1]:
                thresholds = [confidence, quality, novelty]
                action = route(p, q, u, thresholds, c)
                coverage = (action[readable_known] == 0).mean()
                if abs(coverage - c['coverage']) > c['coverage_tolerance']:
                    continue
                retention = [(action[readable_known & (labels == k)] == 0).mean() for k in range(c['classes']) if (readable_known & (labels == k)).any()]
                if c['variant'] != 'no-retention' and min(retention) < c['minimum_retention']:
                    continue
                accepted = (action == 0) & (labels >= 0)
                error = (p.argmax(1)[accepted] != labels[accepted]).mean() if accepted.any() else 1.
                objective = error + c['reacquisition_cost'] * (action == 1).mean() + c['review_cost'] * (action == 2).mean()
                if objective < best:
                    best, selected = objective, thresholds
    (out / 'policy.json').write_text(json.dumps(dict(feasible=selected is not None, thresholds=selected)))
    np.savez(out / 'novelty.npz', centers=centers, scales=scales, mean=standardizer.mean_, scale=standardizer.scale_,
             coefficient=learner.coef_[0], intercept=learner.intercept_[0])


def run(c, out):
    policy = json.loads((out / 'policy.json').read_text(encoding='utf-8'))
    if not policy['feasible']:
        raise ValueError('No calibration policy satisfies coverage and class-retention constraints')
    d = np.load(out / 'data.npz')
    fitted = np.load(out / 'novelty.npz')
    model = Triage(d['x'].shape[1], c)
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True, map_location='cpu'))
    test = d['split'] == 'test'
    p, z, q, energy = predict(model, d['x'][test], c)
    feature = novelty_features(z, q, energy, fitted['centers'], fitted['scales'], c)
    logit = ((feature - fitted['mean']) / fitted['scale']) @ fitted['coefficient'] + fitted['intercept']
    u = 1 / (1 + np.exp(-np.clip(logit, -700, 700)))
    actions = route(p, q, u, policy['thresholds'], c)
    np.savez_compressed(out / 'predictions.npz', probability=p, defect=q, novelty=u, action=actions,
                        y=d['y'][test], readable=d['readable'][test], technical=d['defect'][test])


def evaluate(c, out):
    d = np.load(out / 'predictions.npz')
    p, y, actions = d['probability'], d['y'], d['action']
    readable = d['readable'].astype(bool)
    known = (y >= 0) & readable
    accepted = actions == 0
    correct = p.argmax(1) == y
    retention = [(actions[known & (y == k)] == 0).mean() for k in range(c['classes']) if (known & (y == k)).any()]
    target_action = np.where(d['technical'] > 0, 1, np.where(y < 0, 2, 0))
    result = dict(method=c['variant'], clean_macro_f1=f1_score(y[known & (d['technical'] == 0)], p[known & (d['technical'] == 0)].argmax(1), average='macro', zero_division=0),
                  semantic_auroc=roc_auc_score(y[readable] < 0, d['novelty'][readable]),
                  accepted_error=float((~correct[accepted & (y >= 0)]).mean()) if (accepted & (y >= 0)).any() else np.nan,
                  accepted_unknown_rate=float(accepted[y < 0].mean()) if (y < 0).any() else np.nan,
                  readable_known_coverage=float(accepted[known].mean()), minimum_class_retention=float(min(retention)),
                  wrong_action_rate=float((actions != target_action).mean()), reacquisition_rate=float((actions == 1).mean()),
                  expert_review_rate=float((actions == 2).mean()))
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)
    pd.crosstab(pd.Series(target_action, name='target'), pd.Series(actions, name='action')).to_csv(out / 'action_confusion.csv')


def plot(out):
    d = np.load(out / 'predictions.npz')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d['defect'], d['novelty'], c=d['action'], s=6, cmap='viridis')
    ax.set(xlabel='Technical defect score', ylabel='Conditioned novelty')
    fig.tight_layout()
    fig.savefig(out / 'routing.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'calibrate', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'no-quality', 'no-retention', 'no-consistency', 'collapsed-actions']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
