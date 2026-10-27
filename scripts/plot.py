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


def diagnostic(run, output):
    """Rebuild source diagnostic figures solely from retained numeric evidence."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = json.loads((run/'train/metrics.json').read_text(encoding='utf-8'))
    dev = json.loads((run/'dev/metrics.json').read_text(encoding='utf-8'))
    points = json.loads((run/'workpoints.json').read_text(encoding='utf-8'))
    protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    names = list(points)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(run/'train/pr.npz', allow_pickle=False) as tr, np.load(run/'dev/pr.npz', allow_pickle=False) as dv:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, name in zip(axes.flat, names):
            for curves, label, color in [(tr, 'Train native AP PR', '#2878b5'), (dv, 'Dev native AP PR', '#d55e00')]:
                p, r = curves[name+'_ap_precision'][:, 0], curves[name+'_ap_recall'][:, 0]
                ax.plot(r, p, label=label, color=color, alpha=.85)
            ax.plot(dv[name+'_work_recall'], dv[name+'_work_precision'], color='#228833',
                    linestyle=':', label='Dev filter/rematch PR')
            point = dev['per_class'][name]['workpoint']
            if point['precision'] is not None:
                ax.scatter(point['recall'], point['precision'], color='black', s=28, zorder=5)
            ax.axhline(protocol['rules']['classes'][name]['precision'], color='#777777', linestyle='--', linewidth=.8)
            ax.set(title=name, xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle('Fixed A best | IoU 0.5 | black: selected dev workpoint; dashed: required precision')
        fig.tight_layout(); fig.savefig(output/'pr.png', dpi=140); plt.close(fig)
    errors = json.loads((run/'errors.json').read_text(encoding='utf-8'))['dev']
    kinds = sorted({k for v in errors.values() for k in v})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottoms = np.zeros(len(names))
    for kind in kinds:
        values = np.array([errors[n].get(kind, 0) for n in names])
        ax.bar(np.arange(len(names))-.15, values, width=.5, bottom=bottoms, label=kind)
        bottoms += values
    fn = [dev['per_class'][n]['workpoint']['fn'] for n in names]
    ax.bar(np.arange(len(names))+.25, fn, width=.2, color='#aaaaaa', label='FN (separate GT side)')
    ax.set(xticks=np.arange(len(names)), xticklabels=names, ylabel='Count',
           title='Dev errors at selected class workpoints | FP types are candidate attributions')
    ax.legend(fontsize=7, ncols=2); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(output/'errors.png', dpi=140); plt.close(fig)
    with (run/'slices.csv').open(encoding='utf-8', newline='') as stream:
        slices = list(csv.DictReader(stream))
    bins = ['width_le8', 'width_8_16', 'width_gt16']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, name in zip(axes.flat, names):
        for purpose, delta, color in [('train', -.18, '#2878b5'), ('dev', .18, '#d55e00')]:
            selected = {r['slice']: r for r in slices if r['purpose'] == purpose and r['class'] == name}
            values = [float(selected[b]['recall50']) if b in selected else np.nan for b in bins]
            bars = ax.bar(np.arange(3)+delta, values, .34, label=purpose, color=color)
            for b, bar in zip(bins, bars):
                if b in selected:
                    ax.text(bar.get_x()+bar.get_width()/2, min(.98, bar.get_height()+.025),
                            'n='+selected[b]['gt'], ha='center', fontsize=7, rotation=90)
        ax.set(title=name, ylim=(0, 1.15), xticks=np.arange(3), xticklabels=['<=8', '(8,16]', '>16'],
               xlabel='GT width at 640 reference (px)', ylabel='Recall @ IoU 0.5')
        ax.grid(axis='y', alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('GT width at 640 reference | fixed dev thresholds applied to train and dev')
    fig.tight_layout(); fig.savefig(output/'sizes.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/a_001'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluation', type=Path, help='Fixed-checkpoint train/dev comparison directory')
    parser.add_argument('--diagnostic', type=Path, help='Retained fixed-checkpoint diagnostic evidence')
    args = parser.parse_args()
    if args.diagnostic:
        diagnostic(args.diagnostic.resolve(), args.output or Path('plots') / args.diagnostic.name)
    elif args.evaluation:
        comparison(args.evaluation.resolve(), args.output or Path('plots') / args.evaluation.name)
    else:
        plot(args.run.resolve(), args.output or Path('plots') / args.run.name)
