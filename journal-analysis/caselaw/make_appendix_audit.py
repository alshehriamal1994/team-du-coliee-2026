#!/usr/bin/env python3
"""Regenerate appendix_audit.tex (Appendix C) from expH_runs.csv.

Rounding is half-up to three decimals, matching the main-text tables. Python's
round() is half-even and produced .423/.294/.284 where the main tables carry
.424/.295/.285.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).parent

def d3(x):
    return str(Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)).lstrip('0')

def d2(x):
    return str(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def esc(s):
    return s.replace('_', r'\_').replace('&', r'\&')

rows = list(csv.DictReader(open(HERE / 'expH_runs.csv')))
out = [
 r'\begin{longtable}{@{}llrrrr@{}}',
 r'\caption{Answer-set sizes of every scored official run, derived from published precision and',
 r'recall together with each split gold total and query count. Zero-score runs are excluded, ten',
 r'in Task~1 and three in Task~2. Sizes are averages per query and may be compared with the gold',
 r'means of 4.38 and 2.94.}\label{tab:appaudit}\\',
 r'\toprule',
 r'run & team & P & R & F1 & size \\ \midrule',
 r'\endfirsthead',
 r'\toprule run & team & P & R & F1 & size \\ \midrule \endhead',
 r'\bottomrule \endfoot',
]
for task, title in (('T1', 'Task 1'), ('T2', 'Task 2')):
    out.append(r'\multicolumn{6}{l}{\emph{%s}} \\' % title)
    for r in [x for x in rows if x['task'] == task]:
        out.append('%s & %s & %s & %s & %s & %s \\\\' % (
            esc(r['run']), esc(r['team']),
            d3(r['P']), d3(r['R']), d3(r['F1']), d2(r['avg_set_size'])))
out.append(r'\end{longtable}')
(HERE / 'appendix_audit.tex').write_text('\n'.join(out) + '\n')
print('wrote appendix_audit.tex:', len(rows), 'runs')
