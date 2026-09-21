"""Single diagram of the IL1-dependent plane shift, from verified saved values."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
settings = json.loads((folder/'IL1_object_plane_defocus.json').read_text())['settings']
geometry = json.loads((folder/'objective_image_plane.json').read_text())
colors = ['#31688e', '#35a779', '#de9b24', '#b54b72']
fig, ax = plt.subplots(figsize=(8, 6.5))
fig.subplots_adjust(left=.15, right=.96, top=.88, bottom=.11)
ax.set_title('Plane shift with IL1', fontsize=21, weight='bold', pad=18)
ax.axvline(.43, color='#d3d8df', lw=1, zorder=0)
for rec, color in zip(settings[1:], colors):
    y = rec['plane_shift_mm']
    ax.hlines(y, .1, .75, color=color, lw=2.5)
    ax.text(.81, y, f"IL1 {rec['IL1_delta']:+.0%}   |   {y:+.2f} mm",
            va='center', fontsize=13, color=color)
ax.hlines(0, .1, .75, color='#7c3aed', lw=1.8, ls=':')
ax.text(.81, -1.0, 'Objective image · reference', va='bottom', fontsize=12, color='#7c3aed')
z_il1 = geometry['distance_before_IL1_m']*1000
ax.hlines(z_il1, .1, .75, color='#334155', lw=5)
ax.text(.81, z_il1, 'IL1 lens', va='center', fontsize=13, color='#334155')
ax.set(xlim=(0, 1.7), ylim=(32, -62), xticks=[], yticks=[-50, -25, 0, 25],
       ylabel='Shift from objective image (mm)')
ax.yaxis.label.set_size(13)
ax.tick_params(axis='y', labelsize=12, length=4, color='#94a3b8')
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.spines['left'].set_color('#94a3b8')
ax.text(.43, -58, 'Towards objective', ha='center', va='center', fontsize=11, color='#64748b')
for suffix in ['png', 'svg', 'pdf']:
    fig.savefig(folder/'figures'/f'03e_plane_shift_only.{suffix}', dpi=200, facecolor='white')
print('Saved one plane-shift diagram in PNG, SVG and PDF.')
