"""Publication-quality figures, v2. Ledger-driven, greyscale-safe ink+accent palette
(CVD dE 24.5, secondary encoding everywhere two series appear). 2026-07-22."""
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).parent
INK, ACC, LIGHT, MUT = '#333333', '#4477aa', '#c9c9c9', '#666666'
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#999999', 'xtick.color': '#666666', 'ytick.color': '#666666',
    'axes.labelcolor': '#333333', 'axes.titlecolor': '#333333',
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e8e8e8',
    'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.dpi': 300, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.06,
})

expA = json.load(open(HERE / 'expA_numbers.json'))
expA2 = json.load(open(HERE / 'expA2_numbers.json'))
expB = json.load(open(HERE / 'expB_numbers.json'))
expC = json.load(open(HERE / 'expC_numbers.json'))
expD = json.load(open(HERE / 'expD_numbers.json'))
expDr = json.load(open(HERE / 'expD_r1_numbers.json'))
t2h = {int(k): v for k, v in json.load(open(HERE / 'fig_data_t2_counts.json'))['test_gold_count_histogram'].items()}

# ── Figure: answer-count distributions ──
fig, (a, b) = plt.subplots(1, 2, figsize=(5.14, 2.05), gridspec_kw={'wspace': 0.28})
dist = {int(k): v for k, v in expA['gold_count_distribution'].items()}
a.bar(list(dist), list(dist.values()), color=INK, width=0.82)
a.axvline(4.375, color=ACC, ls=(0, (4, 3)), lw=1.1)
a.text(4.9, 76, 'mean 4.38', color=ACC, fontsize=8)
a.annotate('24.8% of queries\ncite more than five', xy=(9, 11), xytext=(13.5, 42),
           fontsize=8, color=MUT, ha='center',
           arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8,
                           connectionstyle='arc3,rad=-0.2'))
a.set_xlabel('gold citations per query')
a.set_ylabel('queries')
a.set_title('(a)  Task 1 test collection', loc='left')
a.set_xlim(0, 26)
b.bar(list(t2h), list(t2h.values()), color=INK, width=0.6)
b.axvline(2.94, color=ACC, ls=(0, (4, 3)), lw=1.1)
b.annotate('test mean\n2.94', xy=(2.94, 0.86), xycoords=('data', 'axes fraction'),
           xytext=(14, 0), textcoords='offset points', ha='left', va='top',
           fontsize=7.2, color=ACC)
b.axvline(1.22, color='#999999', ls=(0, (2, 2)), lw=1.1)
b.annotate('development\nmean 1.22', xy=(1.22, 0.86), xycoords=('data', 'axes fraction'),
           xytext=(-8, 0), textcoords='offset points', ha='right', va='top',
           fontsize=7.2, color=MUT)
b.set_xlabel('gold paragraphs per case')
b.set_ylabel('cases')
b.set_title('(b)  Task 2 test collection', loc='left')
b.set_xlim(0.3, 6.7); b.set_xticks(range(1, 7))
fig.savefig(HERE / 'fig_counts.pdf'); plt.close(fig)

# ── Figure: evidence depth ECDF ──
offs = sorted(expB['evidence_paragraph']['offsets'])
n = len(offs)
fig, ax = plt.subplots(figsize=(5.14, 2.60))
ax.step(offs, [i / n for i in range(1, n + 1)], color=INK, lw=1.7)
for w, share, lab, side in [(512, .283, '512 tokens\n71.7% beyond', 'right'),
                            (4096, .857, '4,096\n14.3%\nbeyond', 'band'),
                            (8192, .961, '8,192\n3.9% beyond', 'right')]:
    ax.axvline(w, color=ACC, ls=(0, (4, 3)), lw=1.0)
    if side == 'right':
        ax.annotate(lab, xy=(w, share), xytext=(w * 1.18, share - 0.16),
                    fontsize=8, color=ACC, va='top')
    else:
        # stacked inside the 4,096-to-8,192 band so no line cuts the text
        ax.annotate(lab, xy=(w, share), xytext=((w * 8192) ** 0.5, share - 0.16),
                    fontsize=8, color=ACC, va='top', ha='center')
med = expB['evidence_paragraph']['median_token_offset']
ax.plot([med], [0.5], 'o', color=INK, ms=5)
ax.annotate(f'median {med:,}', xy=(med, 0.5), xytext=(med * 0.32, 0.62),
            fontsize=8, color=MUT)
ax.set_xscale('log'); ax.set_xlim(64, 22000); ax.set_ylim(0, 1.02)
ax.set_xticks([100, 512, 1000, 4096, 8192])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xticklabels(['100', '512', '1,000', '4,096', '8,192'])
ax.set_xlabel('token offset of best-evidence paragraph (log scale)')
ax.set_ylabel('cumulative fraction of gold pairs')
fig.savefig(HERE / 'fig_depth.pdf'); plt.close(fig)

# ── Figure: fixed-count curves ──
fig, (a, b) = plt.subplots(1, 2, figsize=(5.14, 2.28), gridspec_kw={'wspace': 0.3})
ceil = {int(k[1:]): v['F1'] for k, v in expA['perfect_system_ceiling'].items()}
ck = sorted(ceil)
a.plot(ck, [ceil[k] for k in ck], 's--', color=ACC, ms=4.5, lw=1.2, mfc='white', mew=1.2)
a.text(3.0, 0.745, 'perfect-ranker ceiling', color=ACC, fontsize=8)
ks = list(range(1, 11))
a.plot(ks, [expA2['fixed_k'][f'k{k}']['F1'] for k in ks], 'o-', color=INK, ms=4.5, lw=1.4)
a.text(4.6, 0.255, 'our ranker\n(post-competition)', color=INK, fontsize=8, ha='center')
a.set_xticks(ks)
a.set_xlabel('fixed answer count $k$'); a.set_ylabel('micro-F1')
a.set_title('(a)  Task 1', loc='left'); a.set_ylim(0.13, 0.80)
kt = list(range(1, 7))
b.plot(kt, [expC['fixed_topk'][f'k{k}']['F1'] for k in kt], 'o-', color=INK, ms=4.5, lw=1.4)
b.text(4.35, 0.384, 'reranker top-$k$', color=INK, fontsize=8, ha='left')
refs = [(0.5076, 'single-selection ceiling  .508', 0.5165),
        (0.4899, 'official winner  .490', 0.4775)]
for y, lab, ty in refs:
    b.axhline(y, color=ACC if 'winner' in lab else '#999999',
              ls=(0, (4, 3)) if 'winner' in lab else (0, (2, 2)), lw=1.0)
    b.text(1.15, ty, lab, fontsize=7.5, va='center',
           color=ACC if 'winner' in lab else MUT)
b.axhline(0.4932, color='#999999', ls=(0, (1, 2)), lw=1.0)
b.text(1.15, 0.4932, 'oracle count  .493', fontsize=7.5, va='bottom', color=MUT)
b.set_xlim(0.6, 6.6); b.set_xticks([k for k in kt if k <= 6]); b.set_ylim(0.30, 0.545)
b.set_xlabel('fixed answer count $k$')
b.set_title('(b)  Task 2', loc='left')
fig.savefig(HERE / 'fig_fixedk.pdf'); plt.close(fig)

# ── Figure: the two-task ladder ──
fig, (a, b) = plt.subplots(1, 2, figsize=(5.14, 2.55),
                           gridspec_kw={'wspace': 0.30, 'width_ratios': [3, 5]})
def ladder(ax, rungs, winner_y, winner_lab, submitted):
    xs = list(range(len(rungs)))
    ax.plot(xs, [v for _, v in rungs], '-', color='#d8d8d8', lw=1.2, zorder=1)
    for i, (lab, v) in enumerate(rungs):
        ax.plot([i], [v], 'o', color=INK, ms=6, zorder=3)
        near = abs(v - winner_y) < 0.03 and v < winner_y + 0.012
        ax.annotate(f'{v:.3f}'.lstrip('0'), (i, v), textcoords='offset points',
                    xytext=(0, -15) if near else (0, 8), ha='center',
                    fontsize=7.5, color=INK)
    si, sv = submitted
    ax.plot([si - 0.32], [sv], 'o', mfc='white', mec=INK, mew=1.3, ms=6, zorder=2)
    ax.annotate('submitted ' + f'{sv:.3f}'.lstrip('0'), (si - 0.32, sv),
                textcoords='offset points', xytext=(4, -12), ha='left',
                fontsize=7, color=MUT)
    ax.axhline(winner_y, color=ACC, ls=(0, (4, 3)), lw=1.1)
    ax.annotate(winner_lab, xy=(len(rungs) - 0.5, winner_y), xytext=(0, 4),
                textcoords='offset points', ha='right', va='bottom',
                fontsize=7.5, color=ACC)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _ in rungs], fontsize=7.2)
    ax.set_xlim(-0.62, len(rungs) - 0.40)
    ax.grid(axis='y')
ladder(a, [('fixed\nfive', 0.3456), ('learned\nstop', 0.3597), ('oracle\ncount', 0.4143)],
       0.4220, 'winner .422', (0, 0.3141))
a.set_ylim(0.28, 0.46); a.set_ylabel('micro-F1'); a.set_title('(a)  Task 1', loc='left', pad=6)
ladder(b, [('top-1', 0.3503), ('top-3', 0.4646), ('oracle', 0.4932),
           ('LLM\nmulti', 0.5487), ('vote', 0.5550)],
       0.4899, 'winner .490', (0, 0.3377))
b.set_ylim(0.28, 0.60); b.set_title('(b)  Task 2', loc='left', pad=6)
fig.savefig(HERE / 'fig_ladder.pdf'); plt.close(fig)

# ── Figure: thresholds flat then collapse ──
fig, (a, b) = plt.subplots(1, 2, figsize=(5.14, 2.20), gridspec_kw={'wspace': 0.28})
ths = sorted(float(t) for t in expC['abs_threshold_min1'])
a.plot(ths, [expC['abs_threshold_min1'][f'{t:.2f}']['F1'] for t in ths],
       'o-', color=INK, ms=4, lw=1.3)
rs = sorted(float(t) for t in expC['rel_threshold_min1'])
a.plot(rs, [expC['rel_threshold_min1'][f'{t:.2f}']['F1'] for t in rs],
       's--', color=ACC, ms=4, lw=1.2, mfc='white', mew=1.1)
a.text(0.51, 0.398, 'absolute', color=INK, fontsize=8)
a.text(0.63, 0.342, 'relative', color=ACC, fontsize=8)
a.axhline(0.4646, color='#999999', ls=(0, (2, 2)), lw=1.0)
a.text(0.5, 0.470, 'fixed $k$=3   .465', fontsize=8, color=MUT)
a.set_ylim(0.30, 0.50)
a.set_xlabel('score threshold'); a.set_ylabel('micro-F1')
a.set_title('(a)  reranker scores', loc='left')
for d, lab, c, m, ls, tx, ty in [(expD, 'DeepSeek-V3', INK, 'o', '-', 0.42, 0.552),
                                 (expDr, 'DeepSeek-R1', ACC, 's', '--', 0.02, 0.462)]:
    ts = sorted(float(t) for t in d['tau_sweep_raw'])
    b.plot(ts, [d['tau_sweep_raw'][f'{t:.2f}']['F1'] for t in ts],
           marker=m, ls=ls, color=c, ms=4, lw=1.3, mfc='white' if m == 's' else c, mew=1.1)
    b.text(tx, ty, lab, color=c, fontsize=8)
b.annotate('91% of selections\nrated 0.8 or higher', xy=(0.8, 0.5401), xytext=(0.13, 0.36),
           fontsize=8, color=MUT,
           arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8,
                           connectionstyle='arc3,rad=0.0'))
b.set_ylim(0.27, 0.60)
b.set_xlabel(r'confidence threshold $\tau$')
b.set_title('(b)  elicited confidence', loc='left')
fig.savefig(HERE / 'fig_thresholds.pdf'); plt.close(fig)

# leaderboard figure now built by make_fig_leaderboard.py
print('v2 figures written')
