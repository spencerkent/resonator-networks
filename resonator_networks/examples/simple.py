"""
Simple demo of simulating a Resonator Network
"""
import _set_the_path

from dynamics import rn_numpy
from utils import encoding_decoding
from utils import plotting

factor_labels = ['Factor1', 'Factor2', 'Factor3']
num_neurons = 1000
cbook_size = 50  # all factors will have same codebook size

num_trials = 5
for _ in range(num_trials):
  the_codebooks = encoding_decoding.generate_codebooks(
      factor_labels, num_neurons, {x: cbook_size for x in factor_labels})
  composite_query, gt_vecs, gt_cbook_indexes = \
      encoding_decoding.generate_c_query(the_codebooks)
  liveplot = plotting.LiveResonatorPlot(
      'sim_bar_plots', {x: gt_vecs[x] for x in gt_vecs}, factor_labels,
      composite_query, the_codebooks)
  decoded_factors, _ , _ = rn_numpy.run(composite_query, the_codebooks,
      synapse_type='OP', live_plotting_obj=liveplot)
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
