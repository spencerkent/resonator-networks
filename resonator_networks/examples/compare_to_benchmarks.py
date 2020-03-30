"""
Simple comparison between Resonator Networks and benchmarks
"""
import _set_the_path

import numpy as np

from dynamics import rn_numpy
from dynamics import bench_numpy
from utils import encoding_decoding
from utils import plotting

factor_labels = ['Factor1', 'Factor2', 'Factor3']
num_neurons = 1000
cbook_size = 35
compare_these_algorithms = ['Alternating Least Squares',
                            'Projected Gradient Descent',
                            'Multiplicative Weights',
                            'Iterative Soft Thresholding',
                            'Fast Iterative Soft Thresholding',
                            'Map Seeking Circuits',
                            'Resonator Networks']
num_trials = 5
show_liveplot = True

def main():
  trial_accs = {x: [] for x in compare_these_algorithms}
  for _ in range(num_trials):
    alg_accs = trial(compare_these_algorithms, with_liveplot=show_liveplot)
    for alg in alg_accs:
      trial_accs[alg].append(alg_accs[alg])

  print('Over', num_trials, 'trials, the accuracy for each alg was:' )
  for alg in compare_these_algorithms:
    print('***', alg, '***')
    print('Mean: ', np.mean(trial_accs[alg]))
    print('Median: ', np.median(trial_accs[alg]))
    print('Variance: ', np.var(trial_accs[alg]))

def trial(which_algs, with_liveplot=False):
  the_codebooks = encoding_decoding.generate_codebooks(
      factor_labels, num_neurons, {x: cbook_size for x in factor_labels})
  composite_query, gt_vecs, gt_cbook_indexes = \
      encoding_decoding.generate_c_query(the_codebooks)
  accs = {}
  for alg in which_algs:
    print('******** Simulating', alg, '********')
    if with_liveplot:
      liveplot = plotting.LiveResonatorPlot(
          'sim_bar_plots', {x: gt_vecs[x] for x in gt_vecs}, factor_labels,
          composite_query, the_codebooks)
    else:
      liveplot = None
    if alg == 'Alternating Least Squares':
      alg_params = {'convergence_tol': 1e-3}
    elif alg == 'Projected Gradient Descent':
      alg_params = {'stepsize': 0.01, 'convergence_tol': 1e-5}
    elif alg == 'Multiplicative Weights':
      alg_params = {'stepsize': 0.3, 'convergence_tol': 1e-3}
    elif alg == 'Iterative Soft Thresholding':
      alg_params = {'sparsity_weight': 0.01, 'convergence_tol': 1e-3}
    elif alg == 'Fast Iterative Soft Thresholding':
      alg_params = {'sparsity_weight': 0.01, 'convergence_tol': 1e-3}
    elif alg == 'Map Seeking Circuits':
      alg_params = {'stepsize': 0.1, 'convergence_tol': 1e-4,
                    'clipping_threshold': 1e-5}

    if alg != 'Resonator Networks':
      decoded_factors, _ = bench_numpy.run(composite_query, the_codebooks,
          algorithm=alg,
          algorithm_params=alg_params,
          live_plotting_obj=liveplot)
    else:
      decoded_factors, _ , _ = rn_numpy.run(composite_query, the_codebooks,
          synapse_type='OP',
          live_plotting_obj=liveplot)

    if liveplot is not None:
      liveplot.ClosePlot()

    best_guesses = encoding_decoding.best_guess(decoded_factors, the_codebooks)
    accuracy = encoding_decoding.calculate_accuracy(best_guesses,
                                                    gt_cbook_indexes)
    print('The best guess based on the final state of the model is:')
    print(best_guesses)
    print('While the ground truth is:')
    print(gt_cbook_indexes)
    print('...for an accuracy of ', accuracy)
    print('---------')
    accs[alg] = accuracy
  return accs

if __name__ == '__main__':
  main()
