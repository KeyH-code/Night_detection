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


def source_pair(evidence, output):
    """Rebuild S3 pair figures from sealed comparison and sampling numbers only."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from source_policy import load_rules
    from source_contract import NAMES
    result = json.loads((evidence/'comparison.json').read_text(encoding='utf-8'))
    runs = result['runs']
    rules = load_rules(Path(runs[0]['run'])/'code/RULES.md')
    output.mkdir(parents=True, exist_ok=True)
    colors = {'R': '#2878b5', 'S': '#d55e00'}
    seeds = ', '.join(str(p['seed']) for p in result['per_seed'])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for arm, offset in [('R', -.18), ('S', .18)]:
        rows = [r for r in runs if r['arm'] == arm]
        ap = [100*np.mean([r[n+'/ap50_95'] for r in rows]) for n in NAMES]
        recall = []
        for n in NAMES:
            valid = all(r[n+'/origin'] == 'rule' and r[n+'/precision'] is not None
                        and r[n+'/precision'] >= rules['classes'][n]['precision'] for r in rows)
            recall.append(100*np.mean([r[n+'/recall'] for r in rows]) if valid else np.nan)
        for ax, values in zip(axes, (ap, recall)):
            bars = ax.bar(np.arange(6)+offset, values, width=.36, color=colors[arm], label=arm)
            for bar, value in zip(bars, values):
                if np.isfinite(value):ax.text(bar.get_x()+bar.get_width()/2, value+1, f'{value:.1f}', ha='center', fontsize=8)
                else:ax.text(bar.get_x()+bar.get_width()/2, 2, 'N/A', ha='center', fontsize=7, rotation=90)
    for ax, title in zip(axes, ('Dev AP50:95 (%)', 'Dev recall at saved threshold satisfying required precision (%)')):
        ax.set(xticks=np.arange(6), xticklabels=NAMES, ylim=(0, 106), title=title)
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].legend();fig.suptitle('Source R/S | training seed(s): '+seeds)
    fig.tight_layout();fig.savefig(output/'metrics.png', dpi=140);plt.close(fig)

    keys = ['pedestrian/ap50_95', 'pedestrian/recall', 'pedestrian/area_S/recall',
            'pedestrian/width_le8/recall', 'pedestrian/small_thin_union/recall']
    labels = ['AP50:95', 'Constrained recall', 'Original small area recall',
              'Width at 640 <=8 px recall', 'Small/thin union recall']
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, key in enumerate(keys):
        v = result['effects'][key]
        if v['mean_delta'] is None:
            ax.text(0, i, 'undefined', va='center', ha='left');continue
        if v['lower'] is not None and v['upper'] is not None:
            ax.hlines(i, v['lower']*100, v['upper']*100, color=colors['S'], linewidth=2)
        ax.scatter(v['mean_delta']*100, i, color=colors['S'], zorder=3)
    ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
    ax.set(yticks=np.arange(len(keys)), yticklabels=labels, xlabel='S minus R (percentage points)',
           title='Pedestrian effects | paired risk-group 95% intervals | fixed selected checkpoints')
    ax.invert_yaxis();ax.grid(axis='x', alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(output/'effects.png', dpi=140);plt.close(fig)

    with (evidence/'sampling.csv').open(encoding='utf-8', newline='') as stream:sample = list(csv.DictReader(stream))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for arm in ('R','S'):
        epochs = sorted({int(r['epoch']) for r in sample if r['arm'] == arm})
        series = [[],[],[]]
        for epoch in epochs:
            rows = [r for r in sample if r['arm'] == arm and int(r['epoch']) == epoch]
            series[0].append(100*np.mean([float(r['target_slots'])/float(r['slots']) for r in rows]))
            series[1].append(np.mean([float(r['unique_images']) for r in rows]))
            series[2].append(np.mean([float(r['augmented_pedestrian_width_le8']) for r in rows]))
        for ax, values in zip(axes, series):ax.plot(epochs, values, label=arm, color=colors[arm])
    off = [int(r['epoch']) for r in sample if r['mosaic_off'].lower() == 'true']
    for ax, title, ylabel in zip(axes, ('Target-layer anchor slots', 'Distinct anchor images per epoch',
                               'Final augmented pedestrian GT with width <=8 input pixels'), ('Slots (%)','Images','GT boxes')):
        if off:ax.axvline(min(off), color='#555555', linestyle='--', label='Mosaic off')
        ax.set(title=title, ylabel=ylabel);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend();axes[-1].set_xlabel('Epoch')
    fig.suptitle('Actual training exposure | final boxes are not per-GT survival rates')
    fig.tight_layout();fig.savefig(output/'sampling.png', dpi=140);plt.close(fig)


def trajectory(evidence, output):
    """Render retained cloud epoch summaries; never reselect or load a model."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (evidence/'trajectory.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fields = [('ped_ap50_95','Pedestrian AP50:95'), ('ped_ap75','Pedestrian AP75'),
              ('ped_recall','Pedestrian recall | required precision'),
              ('union_recall50','Small/thin union recall | required precision')]
    colors = {'R':'#2878b5','S':'#d55e00'}
    for arm in ('R','S'):
        data = sorted([r for r in rows if r['arm']==arm], key=lambda r:int(r['epoch']))
        x = [int(r['epoch']) for r in data]
        selected = [r for r in data if r['cloud_selected_diagnostic'].lower()=='true']
        for ax, (field,title) in zip(axes.flat,fields):
            y = [100*float(r[field]) if r[field] else np.nan for r in data]
            ax.plot(x,y,color=colors[arm],label=arm,linewidth=1.4)
            for r in selected:
                if r[field]:ax.scatter(int(r['epoch']),100*float(r[field]),s=45,marker='D',
                    color=colors[arm],zorder=4,label=f"{arm} diagnostic {r['epoch']}")
            ax.set(title=title,ylabel='Percent',xlabel='Epoch')
    off = [int(r['epoch']) for r in rows if r['mosaic_off'].lower()=='true']
    for ax in axes.flat:
        if off:ax.axvline(min(off),color='#777777',linestyle='--',linewidth=1,label='Mosaic off')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Original 100-epoch cloud dev records | epochs are correlated, not independent repeats')
    fig.tight_layout();output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'trajectory.png',dpi=140);plt.close(fig)



def night_pair(runs, output):
    """Render the paired 340-image operating curves from sealed development predictions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    import detection_metrics as metrics
    from source_contract import check_equal, sha256, verified_file

    pair = runs / 'supp_002'
    receipt = json.loads((pair / 'manifest.json').read_text(encoding='utf-8'))
    comparison_path = verified_file(pair, {'path': 'comparison.json', **receipt['files']['comparison.json']})
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    classes = ((3, 'truck', 'Truck', 28), (4, 'tram', 'Tram', 150))
    arms = (
        ('control', 'expand_010', '300 control', '#334155', 'solid'),
        ('targeted', 'expand_013', '340 targeted', '#b45309', 'dashdot'),
        ('random', 'expand_014', '340 random', '#0369a1', 'dashed'),
    )
    for arm, run_name, label, color, style in arms:
        run = runs / run_name
        manifest = run / 'manifest.json'
        check_equal(f'{run_name} manifest', comparison['seals'][run_name], sha256(manifest))
        relative = '43/dev/night/objects.jsonl.gz'
        record = json.loads(manifest.read_text(encoding='utf-8'))['files'][relative]
        rows = metrics.read_rows(verified_file(run, {'path': relative, **record}))
        check_equal(f'{run_name} development images', 331, len(rows))
        for ax, (class_id, name, _, maximum) in zip(axes, classes, strict=True):
            curve = metrics.work_curve(rows, class_id)
            eligible = (curve['fp'] >= 0) & (curve['fp'] <= maximum)
            feasible = np.zeros(maximum + 1, dtype=int)
            np.maximum.at(feasible, curve['fp'][eligible], curve['tp'][eligible])
            best = np.maximum.accumulate(feasible)
            recorded = comparison['development'][arm]['night'][name]
            target = recorded['target_tp']
            crossing = np.flatnonzero(best >= target)
            check_equal(f'{run_name}/{name} recorded TP', recorded['tp'], int(best[6]))
            check_equal(f'{run_name}/{name} first target crossing',
                        recorded['min_fp_to_target'], int(crossing[0]) if len(crossing) else None)
            ax.step(range(maximum + 1), best, where='post', color=color, linestyle=style,
                    linewidth=2, label=label)
            ax.scatter(6, best[6], color=color, s=28, zorder=4)

    for ax, (_, name, title, maximum) in zip(axes, classes, strict=True):
        target = comparison['development']['control']['night'][name]['target_tp']
        ax.axhline(target, color='#64748b', linestyle=':', linewidth=1.4)
        ax.axvline(6, color='#94a3b8', linestyle=':', linewidth=1.4)
        ax.annotate(f'Target: {target} TP', (6 + maximum * .08, target), xytext=(0, 5),
                    textcoords='offset points', ha='left', color='#334155', fontsize=9)
        ax.set(title=title, xlim=(0, maximum), ylim=(0, None),
               xlabel='Maximum false positives', ylabel='Best true positives')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Night development set | 331 images | operating budget: 6 FP per class',
                 fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9, frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'recall.png', dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/a_001'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluation', type=Path, help='Fixed-checkpoint train/dev comparison directory')
    parser.add_argument('--diagnostic', type=Path, help='Retained fixed-checkpoint diagnostic evidence')
    parser.add_argument('--source-pair', type=Path, help='Retained S3 comparison.json and sampling.csv directory')
    parser.add_argument('--trajectory', type=Path, help='Offline S3 trajectory.csv directory')
    parser.add_argument('--night-pair', type=Path, help='Local runs root with the sealed 340-image comparison')
    args = parser.parse_args()
    if args.night_pair:
        night_pair(args.night_pair.resolve(), args.output or Path('plots/supp_002'))
    elif args.trajectory:
        trajectory(args.trajectory.resolve(), args.output or Path('plots') / args.trajectory.name)
    elif args.source_pair:
        source_pair(args.source_pair.resolve(), args.output or Path('plots') / args.source_pair.name)
    elif args.diagnostic:
        diagnostic(args.diagnostic.resolve(), args.output or Path('plots') / args.diagnostic.name)
    elif args.evaluation:
        comparison(args.evaluation.resolve(), args.output or Path('plots') / args.evaluation.name)
    else:
        plot(args.run.resolve(), args.output or Path('plots') / args.run.name)
