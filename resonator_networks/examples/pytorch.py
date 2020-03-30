"""
Simple demo of simulating a Resonator Network with PyTorch
"""
import _set_the_path

import torch

from dynamics import rn_pytorch
from utils import encoding_decoding
from utils import plotting

factor_labels = ['Factor1', 'Factor2', 'Factor3']
num_neurons = 1000
cbook_size = 50  # all factors will have same codebook size

gpu0 = torch.device('cuda:0')

num_trials = 5
for _ in range(num_trials):
  the_codebooks = encoding_decoding.generate_codebooks(
      factor_labels, num_neurons, {x: cbook_size for x in factor_labels})
  composite_query, gt_vecs, gt_cbook_indexes = \
      encoding_decoding.generate_c_query(the_codebooks)
  # we will put the data on the gpu here. Also notice that the datatype must
  # be float32 simply because PyTorch's casting is a little less automated than
  # NumPy's and we want to avoid unnecessary copies or casts. Also no live
  # plotting because if we're using PyTorch we want this to be FAAASSST
  decoded_factors_pt, _, _ = rn_pytorch.run(
      torch.from_numpy(composite_query).to(gpu0).type(torch.float32),
      {x: torch.from_numpy(the_codebooks[x]).to(gpu0).type(torch.float32)
       for x in the_codebooks}, synapse_type='OP')
  decoded_factors = {x: decoded_factors_pt[x].cpu().numpy().astype('int8')
                     for x in decoded_factors_pt}
  best_guesses = encoding_decoding.best_guess(decoded_factors, the_codebooks)
  accuracy = encoding_decoding.calculate_accuracy(best_guesses,
                                                  gt_cbook_indexes)
  print('The best guess based on the final state of the model is:')
  print(best_guesses)
  print('While the ground truth is:')
  print(gt_cbook_indexes)
  print('...for an accuracy of ', accuracy)
  print('---------')
