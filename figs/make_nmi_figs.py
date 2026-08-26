#!/usr/bin/env python3
"""Generate NMI-quality comparison figures for GNOmE."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__))

BG = '#0b0f14'
FG = '#e7edf3'
ACCENT = '#3b82f6'
ACCENT2 = '#a78bfa'
ACCENT3 = '#f59e0b'
MUTED = '#94a3b8'
CARD = '#111827'
LINE = '#1e293b'
OK = '#4ade80'
BAD = '#f87171'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'text.color': FG, 'axes.labelcolor': FG,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': LINE, 'grid.color': LINE,
    'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 200,
})


# ═══════════════════════════════════════════════
# Figure 1: Query complexity comparison
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

methods = ['Path\npatching', 'ACDC', 'Activation\npatching', 'Attribution\npatching', 'GNOmE\n(Ours)']
queries = [24336, 156*156, 156*156, 156*156, 1]
colors = [BAD, '#f97316', '#eab308', ACCENT3, OK]

bars = ax.bar(methods, np.log10(queries), color=colors, alpha=0.85, width=0.55)
for bar, q in zip(bars, queries):
    label = '1' if q == 1 else f'{q:,}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            label, ha='center', fontsize=11, color=FG, fontweight='bold')

ax.set_ylabel('log₁₀(forward passes required)', color=MUTED)
ax.set_title('Query Complexity: GNOmE vs Intervention-Based Methods\n(GPT-2 Small, 156 units)', fontsize=14, fontweight='bold', color=FG)
ax.set_ylim(0, 5.5)
ax.axhline(y=0, color=LINE, linestyle='--', alpha=0.3)

# Annotate the advantage
ax.annotate('24,336× fewer\nforward passes',
            xy=(4, 0.15), xytext=(2.8, 3.5),
            fontsize=11, color=OK, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=OK, lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD, edgecolor=OK, alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_query_complexity.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("✓ fig_query_complexity.png")


# ═══════════════════════════════════════════════
# Figure 2: Correlation comparison (GNOmE vs path patching)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

methods_corr = ['Path patching', 'GNOmE\n(Ours)']
pearson = [-0.365, 0.748]
colors_corr = [BAD, OK]

bars = ax.bar(methods_corr, pearson, color=colors_corr, alpha=0.85, width=0.4)
ax.axhline(y=0, color=MUTED, linestyle='-', linewidth=0.8)

for bar, val in zip(bars, pearson):
    offset = 0.05 if val >= 0 else -0.12
    ax.text(bar.get_x() + bar.get_width()/2, val + offset,
            f'r = {val:.3f}', ha='center', fontsize=12, color=FG, fontweight='bold')

ax.set_ylabel('Pearson r vs ground truth', color=MUTED)
ax.set_title('Correlation with Zero-Ablation Ground Truth\n(Six 2-layer transformers)', fontsize=14, fontweight='bold', color=FG)
ax.set_ylim(-0.6, 1.0)

# Add a reference line for "good" correlation
ax.axhline(y=0.5, color=ACCENT, linestyle=':', alpha=0.4, linewidth=1)
ax.text(1.15, 0.52, 'r = 0.5 threshold', fontsize=8, color=ACCENT, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_correlation.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("✓ fig_correlation.png")


# ═══════════════════════════════════════════════
# Figure 3: Cross-task transfer heatmap
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))

data = np.array([
    [0.864, 0.954],
    [0.963, 0.864],
])

im = ax.imshow(data, cmap='YlGnBu', vmin=0.8, vmax=1.0, aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['IOI', 'Induction Heads'], fontsize=11)
ax.set_yticklabels(['IOI', 'Induction Heads'], fontsize=11)
ax.set_xlabel('Target task', color=MUTED, fontsize=12)
ax.set_ylabel('Source task', color=MUTED, fontsize=12)
ax.set_title('Cross-Task Transfer (Pearson r)\nTrained on source, tested on target', fontsize=13, fontweight='bold', color=FG)

for i in range(2):
    for j in range(2):
        color = FG if data[i, j] < 0.92 else BG
        label = f'r = {data[i, j]:.3f}' if i != j else f'r = {data[i, j]:.3f}\n(LOO CV)'
        ax.text(j, i, label, ha='center', va='center', fontsize=12, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pearson r', color=MUTED)
cbar.ax.tick_params(colors=MUTED)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_transfer.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("✓ fig_transfer.png")


# ═══════════════════════════════════════════════
# Figure 4: Convergent discovery timeline
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))

events = [
    (2017, 'Network\nDissection', MUTED),
    (2021, 'Transformer\nCircuits', MUTED),
    (2023, 'Path patching\nACDC', ACCENT3),
    (2023, 'GNOmE\n(Ours)', OK),
    (2025, 'Anthropic\nAttribution', '#f87316'),
]

for year, label, color in events:
    ax.scatter(year, 0, s=120, color=color, zorder=5, edgecolors=FG, linewidth=1)
    offset = 0.35 if year != 2023 else (-0.35 if 'GNOmE' in label else 0.35)
    ax.annotate(label, xy=(year, 0), xytext=(year, offset),
                ha='center', va='bottom' if offset > 0 else 'top',
                fontsize=9, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=color, lw=1))

ax.axhline(y=0, color=LINE, linewidth=2, zorder=1)
ax.set_xlim(2016, 2026)
ax.set_ylim(-0.7, 0.7)
ax.set_xlabel('Year', color=MUTED, fontsize=12)
ax.set_title('Convergent Discovery: From Interventions to Structure Reading', fontsize=14, fontweight='bold', color=FG)
ax.get_yaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_timeline.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("✓ fig_timeline.png")


# ═══════════════════════════════════════════════
# Figure 5: IOI component recovery bar chart
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

components = ['Duplicate-\ntoken heads', 'Name-mover\nL10_H0', 'Negative\nname-movers', 'S-inhibition\nL8_H1', 'Induction\nheads']
zq_ranks = [9.3, 15.0, 19.5, 50.0, 79.5]
pt_ranks = [51.7, 5.0, 24.0, 24.0, 133.0]

x = np.arange(len(components))
width = 0.35

bars1 = ax.bar(x - width/2, zq_ranks, width, label='Zero-query (Ours)', color=OK, alpha=0.85)
bars2 = ax.bar(x + width/2, pt_ranks, width, label='Patching ground truth', color=ACCENT2, alpha=0.6)

ax.axhline(y=156*0.25, color=BAD, linestyle='--', linewidth=1, alpha=0.7, label='Top-25% threshold')
ax.set_ylabel('Rank out of 156 units', color=MUTED)
ax.set_title('IOI Component Recovery on GPT-2 Small\n(lower = better, top-25% threshold at rank 39)', fontsize=14, fontweight='bold', color=FG)
ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=9)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=LINE, labelcolor=FG)
ax.set_ylim(0, 150)

# Mark PASS/FAIL
for i, (zq, pt) in enumerate(zip(zq_ranks, pt_ranks)):
    if zq < 156*0.25:
        ax.text(i - width/2, zq + 3, 'PASS', ha='center', fontsize=8, color=OK, fontweight='bold')
    else:
        ax.text(i - width/2, zq + 3, 'FAIL', ha='center', fontsize=8, color=MUTED)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_ioi_recovery.png'), dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("✓ fig_ioi_recovery.png")

print("\nAll 5 GNOmE figures generated.")
