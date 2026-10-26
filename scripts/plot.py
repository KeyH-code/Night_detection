"""Plot loss and AP curves from a retained run; never load images or models."""
import argparse
import csv
import json
from pathlib import Path


def comparison(evaluation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (evaluation / 'comparison.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [r['class'] for r in rows]
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(9, 4))
    train = [100*float(r['train_ap50_95']) for r in rows]
    dev = [100*float(r['dev_ap50_95']) for r in rows]
    left = ax.bar([i-.18 for i in x], train, width=.36, label='Train: fixed best')
    right = ax.bar([i+.18 for i in x], dev, width=.36, label='Dev: existing evaluation')
    ax.bar_label(left, fmt='%.1f', fontsize=8, padding=2)
    ax.bar_label(right, fmt='%.1f', fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 100), ylabel='AP50:95 (%)', title=evaluation.name)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'ap.png', dpi=130); plt.close(fig)


def plot(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (run / 'train/results.csv').open(encoding='utf-8', newline='') as stream:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    output.mkdir(parents=True, exist_ok=True)
    epochs = [r['epoch'] for r in rows]
    scores = [json.loads(line) for line in (run / 'train/source_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    best = max(scores, key=lambda r: r['fitness'])['epoch']
    config = json.loads((run / 'train/audit_initial/config_resolved.json').read_text(encoding='utf-8'))
    mosaic_epoch = config['training']['epochs'] - config['training']['close_mosaic'] + 1
    with plt.rc_context({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, part, title in zip(axes, ('box', 'cls', 'dfl'), ('Box loss', 'Class loss', 'DFL loss')):
            for prefix in ('train', 'val'):
                ax.plot(epochs, [r[f'{prefix}/{part}_loss'] for r in rows], label=prefix)
            ax.axvline(best, color='#555555', linestyle=':', alpha=.7, label=f'Best: {best}')
            ax.axvline(mosaic_epoch, color='#888888', linestyle='--', alpha=.5, label='Mosaic off')
            ax.set(xlabel='Epoch', ylabel='Loss', title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(run.name)
        fig.tight_layout(pad=1.6); fig.savefig(output / 'loss.png'); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
        for ax, classes, title in zip(axes, (('pedestrian', 'car'), ('cyclist', 'truck', 'tram', 'tricycle')),
                                      ('Primary classes', 'Auxiliary classes')):
            for name in classes:
                ax.plot(epochs, [r['per_class_AP50-95'][name] for r in scores], label=name)
            ax.axvline(best, color='#555555', linestyle=':', alpha=.7)
            ax.axvline(mosaic_epoch, color='#888888', linestyle='--', alpha=.5)
            ax.set(xlabel='Epoch', ylabel='Source dev AP50:95', ylim=(0, 1), title=title)
            ax.grid(alpha=.2)
        axes[0].plot(epochs, [r['fitness'] for r in scores], color='black', linestyle='--', label='Primary macro')
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        fig.suptitle(f'{run.name} | dotted: best {best}; vertical dashed: mosaic off')
        fig.tight_layout(pad=1.6); fig.savefig(output / 'ap.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/a_001'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluation', type=Path, help='Fixed-checkpoint train/dev comparison directory')
    args = parser.parse_args()
    if args.evaluation:
        comparison(args.evaluation.resolve(), args.output or Path('plots') / args.evaluation.name)
    else:
        plot(args.run.resolve(), args.output or Path('plots') / args.run.name)
