import numpy as np
from rom_builder import RomBuilder
from gmpe_analytic import get_analytic
import matplotlib.pyplot as plt

min_vals = [0.0, 0.0, 10.0, 10.0, 0.0]
max_vals = [60.0, 360.0, 50.0, 50.0, 90.0]
n_vals_grid = [5, 5, 5, 5, 5]
npoints_initial = 1000
desired_err = 1e10
n_forwards = 3000
n_err_update = 500
n_samples_refine = 500


plot_vals = [[0, 0], [0, 90], [30, 0]]


def test_dimensions():
    for dim in range(2, 5):
        min_vals_dim = min_vals[0:dim]
        max_vals_dim = max_vals[0:dim]
        n_vals_grid_dim = n_vals_grid[0:dim]
        rb = RomBuilder(
            get_analytic, min_vals_dim, max_vals_dim, n_vals_grid_dim,
            npoints_initial, 'halton', n_samples_refine, desired_err,
            n_forwards, n_err_update, rbf_type='polyh', k_val=5)
        rb.run_training_loop()
        plt.plot(rb.cum_nforwards_err, rb.error_list, '-o', label=dim)
    plt.yscale('log')
    plt.xlabel('Number of forward models')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('errors_vs_n_dimensions.pdf')
    plt.close('all')


def test_refinements():
    for refinement in ['voronoi_vertex', 'voronoi_edge_center', 'halton',
                       'voronoi_random_walk']:
        print(refinement)
        rb = RomBuilder(
            get_analytic, min_vals, max_vals, n_vals_grid, npoints_initial,
            refinement, n_samples_refine, desired_err, n_forwards,
            n_err_update, rbf_type='polyh', k_val=5)
        rb.run_training_loop()
        plt.plot(rb.cum_nforwards_err, rb.error_list, '-o', label=refinement)
    plt.yscale('log')
    plt.xlabel('Number of forward models')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('errors_vs_n_refinements.pdf')
    plt.close('all')


def test_neural_nets():
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for hidden_layer in [1, 3, 5]:
        for n_neurons in [10, 20, 50]:
            rb = RomBuilder(
                get_analytic, min_vals, max_vals, n_vals_grid, npoints_initial,
                'halton', n_samples_refine, desired_err, n_forwards,
                n_err_update, ml_type='nn', epochs=1000, normalize=True,
                hidden_layers=hidden_layer, n_neurons_per_layer=n_neurons)
            rb.run_training_loop()
            ax.plot(rb.cum_nforwards_err, rb.error_list, '-o',
                    label='%s_%s' % (hidden_layer, n_neurons))

            ax2.plot(rb.pod.interpolant.train_history.history['val_loss'],
                     label='%s_%s' % (hidden_layer, n_neurons))

    ax2.set_yscale('log')
    ax2.legend()
    fig2.savefig('compare_loss.pdf')

    ax.set_yscale('log')
    ax.set_xlabel('Number of forward models')
    ax.set_ylabel('Error')
    ax.legend()
    fig.savefig('compare_neural_nets.pdf')


def test_interpolants():
    # Testing interpolants

    # Array for storing predicted PGVs
    # pgvs = np.zeros((5, 3, 2500))

    # interpolants = ['polyh', 'knn', 'dt', 'gpr']
    interpolants = ['polyh', 'lasso']

    fig, ax = plt.subplots()
    for interp_idx, interpolant in enumerate(interpolants):
        print(interpolant)
        if interpolant == 'polyh':
            rbf_type = 'polyh'
            ml_type = None
        else:
            rbf_type = None
            ml_type = interpolant
        rb = RomBuilder(
            get_analytic, min_vals, max_vals, n_vals_grid, npoints_initial,
            'halton', n_samples_refine, desired_err, n_forwards,
            n_err_update, rbf_type=rbf_type, ml_type=ml_type)
        rb.run_training_loop()
        ax.plot(rb.cum_nforwards_err, rb.error_list, '-o', label=interpolant)

        # for plot_idx, plot_val in enumerate(plot_vals):
        #     eval = rb.pod.evaluate_multi(np.array([plot_val]))[0]
        #     pgvs[interp_idx + 1, plot_idx] = eval
        #     pgvs[0, plot_idx] = get_analytic(
        #         plot_val[0], plot_val[1]).flatten()

    ax.set_yscale('log')
    ax.set_xlabel('Number of forward models')
    ax.set_ylabel('Error')
    ax.legend()
    fig.savefig('errors_vs_n_interpolants.pdf')

    # fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))
    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         ax = axes[i, j]
    #         ax.imshow(pgvs[j, i].reshape(50, 50))
    #     # TODO: set titles and ylabels
    #     # TODO: check that pgvs[j, i] is correct
    #         if i == 0:
    #             if j == 0:
    #                 ax.set_title('Truth')
    #             else:
    #                 ax.set_title(interpolants[j-1])
    #         if j == 0:
    #             ax.set_ylabel('depth: %s, strike: %s' %
    #                           (plot_vals[i][0], plot_vals[i][1]))

    # plt.tight_layout()
    # fig.savefig('pgvs_100.pdf')


def test_ranks():
    for interpolant in ['knn', 'dt', 'nn']:
        for rank in [10, 50, None]:
            print(interpolant, rank)
            rb = RomBuilder(
                get_analytic, min_vals, max_vals, n_vals_grid, 100,
                'halton', n_samples_refine, desired_err, n_forwards,
                n_err_update, ml_type=interpolant, rank=rank, epochs=500)
            rb.run_training_loop()
            plt.plot(rb.cum_nforwards_err, rb.error_list, '-o',
                     label='%s_%s' % (interpolant, rank))
    rb.run_training_loop()

    # plt.plot(rb.pod.interpolant.train_history.history['val_loss'],
    #          label='val_loss')
    # plt.plot(rb.pod.interpolant.train_history.history['loss'],
    #          label='loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('learning.pdf')
    # plt.close('all')

    plt.yscale('log')
    plt.xlabel('Number of forward models')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('errors_vs_n_interpolants_ranks.pdf')
    plt.close('all')


if __name__ == '__main__':
    test_dimensions()
    # test_refinements()
    # test_interpolants()
    # test_ranks()
    # test_neural_nets()
