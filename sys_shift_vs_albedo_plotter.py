import matplotlib.pyplot as plt
import numpy as np

albedos = [0.0, 0.2, 0.4, 0.6, 0.8]

data = {
    'U-235 fission': [-0.011204, -0.140324, -0.185883, -0.057979, -0.023452],
    'U-235 capture': [0.221713, -0.082896, 0.071050, 0.230992, 0.165382],
    'U-235 elastic': [0.003803, -0.045671, -0.052845, -0.146771, -0.090221],
    'U-235 inelastic': [-0.629249, -1.021222, -0.808104, -0.906366, 0.306358],
    'U-238 fission': [0.064934, -0.128943, 0.023219, 0.029102, 0.018710],
    'U-238 capture': [0.027829, -0.020922, -0.033601, -0.076352, -0.026268],
    'U-238 elastic': [0.019297, -0.162608, -0.250657, 0.159957, 0.066958],
    'U-238 inelastic': [-0.911376, -0.198162, 0.339911, 0.252836, 0.090259],
    'nu U-235': [-0.056079, -0.050848, -0.047796, -0.042005, -0.030730],
    'nu U-238': [-0.002538, -0.002229, -0.002133, -0.002166, -0.001417],
}

# Increased figure size to accommodate the massive text/legend without overlapping
fig, ax = plt.subplots(figsize=(16, 10))

markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i, ((param, values), marker) in enumerate(zip(data.items(), markers)):
    # Doubled linewidth and markersize so they match the huge text proportion
    ax.plot(albedos, values, label=param, marker=marker, color=colors[i], linewidth=4, markersize=14)

# Doubled the thickness of the zero-line
ax.axhline(0, color='black', linewidth=2, linestyle='--')

# All font sizes perfectly doubled
ax.set_title('Systematic Shift (\u0394R) vs. Reflector Albedo', fontsize=32)
ax.set_xlabel('Reflector Albedo', fontsize=28)
ax.set_ylabel('\u0394R (cm)', fontsize=28)
ax.set_xticks(albedos)
ax.tick_params(axis='both', which='major', labelsize=24)

# Legend font size perfectly doubled, kept outside the plot area
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=24)
ax.grid(True, linestyle=':', alpha=0.7)

# Added bbox_inches='tight' so the giant legend doesn't get cut off when rendering
fig.tight_layout()
plt.savefig('delta_R_vs_albedo_large.png', dpi=300, bbox_inches='tight')
