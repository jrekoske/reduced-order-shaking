fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
cols = ['vp', 'vs', 'rho']
labels = ['$V_p$ (m/s)', '$V_s$ (m/s)', 'Density (kg/m$^3$)']
for i, ax in enumerate(axes):
    ax.hist(df[cols[i]], density=True, edgecolor='k', linewidth=0.5)
    ax.set_xlabel(labels[i])
    ax.set_ylabel('Probability')
plt.tight_layout()
plt.savefig('material_dists.pdf')
plt.show()
