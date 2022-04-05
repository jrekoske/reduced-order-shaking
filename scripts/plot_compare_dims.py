import matplotlib.pyplot as plt
from new_rom_builder import RomBuilder
from gmpe_analytic import get_analytic
plt.rcParams['font.size'] = 13

func = get_analytic
lower_bounds = [0, 0, 15, 0]
upper_bounds = [60, 360, 45, 90]
n_truth = 5
n_samps_initial = 10
n_samps_refine = 100
n_samps_stop = 1000

plt.figure(figsize=(4, 3.5))
for i in range(len(lower_bounds)):
    lower_bounds_dim = lower_bounds[0:i+1]
    upper_bounds_dim = upper_bounds[0:i+1]
    rb = RomBuilder(func, lower_bounds_dim, upper_bounds_dim,
                    n_truth, n_samps_initial, n_samps_refine, n_samps_stop)
    plt.plot(rb.nsamples_list, rb.error_list, '-o', label=i+1)
plt.xlabel('n')
plt.ylabel('error')
plt.yscale('log')
plt.legend()
plt.savefig('pres/figs/errorrs_dim.pdf')
