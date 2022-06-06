import os
import numpy as np
from halton import get_halton
from gmpe_analytic import get_analytic
import pod as podtools
from sklearn.model_selection import KFold
from voronoi import bounded_voronoi, plot_voronoi_diagram
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm

save_dir = 'voronoi_images'
plot_VD = False

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load array containing pre-computed forward models for evaluating errors
shakes = np.load('shakes.npy')

# Create initial forward models
n = 10  # Number of initial forwards
dep_min = 0
dep_max = 60  # km
strike_min = 0
strike_max = 360  # degrees
deps_total = get_halton(dep_min, dep_max, 3, 100)
strikes_total = get_halton(strike_min, strike_max, 4, 100)

# Grid variables for evaluating ROM accuracy on the grid
deps = np.linspace(dep_min, dep_max, 100)
strikes = np.linspace(strike_min, strike_max, 100)
dep_grid, strike_grid = np.meshgrid(deps, strikes)
grid = np.column_stack((dep_grid.ravel(), strike_grid.ravel()))

deps = deps_total[:10]
strikes = strikes_total[:10]

current_halton_i = 10

params = np.vstack((deps, strikes)).T
params_h = np.vstack((deps, strikes)).T

# Run analytic forward models
snapshots = []
controls = {'depth': [], 'strike': []}
snapshots_h = []
controls_h = {'depth': [], 'strike': []}

n_forwards = []
n_forwards_h = []
n_forwards_true = []
max_errors = []
max_errors_h = []

# List of values to try for k-fold cross validation
k_values = [2, 5, 10]

niter = 31

k_fold_max_errors = np.zeros((niter, len(k_values)))

# refinement iterations
for iter_val in tqdm(range(niter)):

    print(iter_val)
    print('Voronoi')

    # Collect new forward models
    print('Collecting forward models')
    for dep, strike in params:
        snapshots.append(get_analytic(dep, strike))
        controls['depth'].append(dep)
        controls['strike'].append(strike)

    n_forwards.append(len(snapshots))

    kfold_errors = np.zeros((len(k_values), len(snapshots)))

    print('Running k-fold')
    for k_idx, k_val in enumerate(k_values):
        print('k=%s' % k_val)
        kf = KFold(n_splits=k_val, shuffle=True)
        for train, test in kf.split(snapshots):
            # Create ROMs
            train_controls = {
                'depth': list(np.array(controls['depth'])[train]),
                'strike': list(np.array(controls['strike'])[train])}
            train_snaps = list(np.array(snapshots)[train])

            pod = podtools.PODMultivariate(remove_mean=True)
            pod.database_append(train_controls, train_snaps)
            pod.setup_basis()
            pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

            # Evaluate errors
            for test_val in test:
                pred = pod.evaluate([controls['depth'][test_val],
                                    controls['strike'][test_val]])[:, 0]

                # TODO: which error to use? Relative? Linf?
                kfold_errors[k_idx][test_val] = np.linalg.norm(
                    snapshots[test_val] - pred) / np.linalg.norm(
                        snapshots[test_val])

    k_fold_max_errors[iter_val] = kfold_errors.max(axis=1)

    if iter_val % 5 == 0:
        print('Getting grid errors')
        # Create ROM using all controls/snapshots
        pod = podtools.PODMultivariate(remove_mean=True)
        pod.database_append(controls, snapshots)
        pod.setup_basis()
        pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

        pred = pod.evaluate_multi(grid).T
        # compute errors from grid evaluation
        max_errors.append(
            max(np.linalg.norm(shakes - pred, axis=1) / np.linalg.norm(
                shakes, axis=1)))
        n_forwards_true.append(len(snapshots))

    print('constructing VD')
    # Construct Voronoi diagram
    all_params = np.array((controls['depth'], controls['strike'])).T
    points = all_params / np.array([dep_max, strike_max])
    bounding_box = np.array([0., 1., 0., 1.])
    vor = bounded_voronoi(points, bounding_box)

    # Define which errors to be used for the Vornoi diagram refinement
    errors_v = kfold_errors[1]

    vertices = vor.vertices[vor.filtered_regions[np.argmax(errors_v)]]
    vertices = vertices[(vertices.T[0] >= 0) & (vertices.T[0] <= 1)]
    vertices = vertices[(vertices.T[1] >= 0) & (vertices.T[1] <= 1)]

    if plot_VD:
        print('Plotting VD')
        norm = mpl.colors.Normalize(vmin=np.min(
            errors_v), vmax=np.max(errors_v), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        fig = plot_voronoi_diagram(vor, bounding_box, False)
        sc = plt.scatter(points.T[0], points.T[1], c=errors_v, zorder=5)
        plt.scatter(points.T[0], points.T[1], c='k', zorder=5)
        plt.scatter(vertices.T[0], vertices.T[1], c='r', zorder=5)
        plt.colorbar(sc, label='Relative $L_2$ error')
        for r, region in enumerate(vor.filtered_regions):
            if -1 not in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=mapper.to_rgba(errors_v[r]))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Depth (normalized)')
        plt.ylabel('Strike (normalized)')
        plt.title('$N=%s$' % len(snapshots))
        plt.savefig(os.path.join(save_dir, 'vor_%s.png' % iter_val))
        plt.close('all')

    # Apply denormalization of vertices
    params = vertices * np.array([dep_max, strike_max])

    n_new_params = len(params)

    # ------------------------------------------ #
    print('Same thing with Halton')
    # Using halton
    for dep, strike in params_h:
        snapshots_h.append(get_analytic(dep, strike))
        controls_h['depth'].append(dep)
        controls_h['strike'].append(strike)

    if iter_val % 5 == 0:
        print('Getting grid errors')
        # Create ROM using all controls/snapshots
        pod = podtools.PODMultivariate(remove_mean=True)
        pod.database_append(controls_h, snapshots_h)
        pod.setup_basis()
        pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

        pred = pod.evaluate_multi(grid).T
        # compute errors from grid evaluation
        max_errors_h.append(
            max(np.linalg.norm(shakes - pred, axis=1) / np.linalg.norm(
                shakes, axis=1)))
        n_forwards_h.append(len(snapshots_h))
        print(len(snapshots_h))

    # Get next halton sequence parameters
    new_params_halton_dep = deps_total[
        current_halton_i:current_halton_i+n_new_params]
    new_params_halton_strike = strikes_total[
        current_halton_i:current_halton_i+n_new_params]
    params_h = np.vstack(
        (new_params_halton_dep, new_params_halton_strike)).T

# Plot comparing VD vertex and Halton sampling
plt.plot(n_forwards_true, max_errors, label='VD vertex')
plt.scatter(n_forwards_true, max_errors)
plt.plot(n_forwards_h, max_errors_h, label='Halton')
plt.scatter(n_forwards_h, max_errors_h)
plt.yscale('log')
plt.savefig('halton_vs_vd_vertex.pdf')
plt.close('all')

# Compare k-fold errors (various values of k) against the "true" error
for kiter, kerr in enumerate(k_fold_max_errors.T):
    plt.scatter(n_forwards, kerr, s=10)
    plt.plot(n_forwards, kerr, label='$k=%s$' % k_values[kiter])
plt.scatter(n_forwards_true, max_errors, s=10)
plt.plot(n_forwards_true, max_errors, label='True error')
plt.yscale('log')
plt.xlabel('Number of forward models')
plt.ylabel('Max relative $L_2$ error')
# plt.ylim(top=1e3)
plt.legend()
plt.savefig('errors_vs_n.pdf')
plt.close('all')
