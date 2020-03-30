"""
NumPy implementation of (discrete-time, bipolar) Resonator Networks

Neuron states are +/- 1, so could be represented in boolean or 8-bit integer
format. However, the dynamics involve summation that will overflow 8 bits,
so in order to avoid recasting in each iteration we represent the state with
32-bit floats.
"""
import copy
import numpy as np

from utils.encoding_decoding import cosine_sim

def run(composite_vec, factor_codebooks, synapse_type='OP', state_init=None,
        max_num_iters=10000, lim_cycle_detection_len=0,
        live_plotting_obj=None, silent=False):
  """
  Factors composite_vec based on the vectors in factor_codebooks

  Parameters
  ----------
  composite_vec : ndarray(int8, size=(N,))
      The N-dimensional vector that we would like to factor
  factor_codebooks : dictionary
      Keys index the label of the factor (could be 0, 1, 2, ... or 'color',
      'shape' etc.) with the dictionary values being matrices whose columns
      are the codebook vectors for that factor.
  synapse_type : str, optional
      One of {'OLS', 'OP'}, specifies whether the synaptic weight matrices
      are computed from the {O}rdinary {L}east {S}quares or the {O}uter
      {P}roduct rule. Default 'OP'.
  state_init : dictionary, optional
      We may want to manually specify the initial state of each factor, in
      which case we can do that here. Each value is a 1d array giving the
      initial state for the factor. Default None.
  max_num_iters : int, optional
      This model is NOT GUARANTEED TO CONVERGE. We can optionally stop it after
      this many discrete-time iterations. In the normal operating regime, this
      may be just a few tens or hundreds of iterations, but depending on how
      large the problem is, can be much larger. If the model is not finding the
      correct factorization, try giving it more iterations. Default 10000.
  lim_cycle_detection_len : int, optional
      The dynamics can converge to a limit cycle, in which case it is
      convenient to detect this and terminate. Looking for limit cycles in the
      dynamics requires extra memory and a slight computational overhead, so
      we turn this off by default. Limit cycle lengths can be arbitrarily long,
      although in practice they are relatively short, (< 20). Limit cycles are
      more common in 2 and 3 factor Resonator Networks. Default 0.
  live_plotting_obj : ResonatorLivePlot, optional
      Used to display updates to the resonator in a Matplotlib window.
      This is the MOST BASIC way to do this and probably won't run any
      faster than about 10 frames a second. Other solutions exit.
  silent : bool, optional
      If True, supress all print statements (useful for large-scale
      simulation). Default False.

  Returns
  -------
  factor_states : dictionary
      Keys index the label of the factor (could be 0, 1, 2, ... or 'color',
      'shape' etc.) and dict values give the final state for that factor.
      States, which are +/- 1, have been recast to int8.
  num_iters : int
      Number of iterations taken by dynamics until either convergence, limit
      cycle detected, or max_num_iters reached.
  lim_cycle_return : dictionary
      Indicates whether or not limit cycles were detected.
      'found' : bool, 'cycle_lengths': dict, optional
  """
  assert composite_vec.dtype == 'int8'
  assert synapse_type in ['OLS', 'OP']
  if state_init is not None:
    assert all([x in factor_codebooks for x in state_init])

  factor_states = dict.fromkeys(factor_codebooks)
  codebook_pseudoinverse = dict.fromkeys(factor_codebooks)
  limit_cycle_detectors = dict.fromkeys(factor_codebooks)
  factor_ordering = list(factor_codebooks.keys())  # update in consistent order
  for factor_label in factor_ordering:
    assert factor_codebooks[factor_label].dtype == 'int8'
    if state_init is not None:
      factor_states[factor_label] = np.copy(
          state_init[factor_label]).astype('float32')
    else:
      factor_states[factor_label] = activation(
          np.sum(factor_codebooks[factor_label], axis=1).astype('float32'))

    if synapse_type == 'OLS':
      codebook_pseudoinverse[factor_label] = np.linalg.pinv(
          factor_codebooks[factor_label])

    if lim_cycle_detection_len > 1:
      limit_cycle_detectors[factor_label] = LimitCycleCatcher(
          len(composite_vec), max_lim_cycle_len=lim_cycle_detection_len)

  if live_plotting_obj is not None:
    live_plotting_obj.UpdatePlot(factor_states)

  iter_idx = 0
  converged = False
  limit_cycle_found = False
  while not converged and not limit_cycle_found and iter_idx < max_num_iters:
    previous_states = copy.deepcopy(factor_states)
    factor_converged = []  # convergence of each factor individually
    factor_has_limit_cycle = []  # do the same for limit cycles
    for factor_label in factor_ordering:
      product_other_factors = np.product(np.array([factor_states[x]
        for x in factor_states if x != factor_label]), axis=0)

      if synapse_type == 'OLS':
        factor_states[factor_label] = activation(
            np.dot(factor_codebooks[factor_label],
                   np.dot(codebook_pseudoinverse[factor_label],
                          composite_vec * product_other_factors)))
      else:
        factor_states[factor_label] = activation(
            np.dot(factor_codebooks[factor_label],
                   np.dot(factor_codebooks[factor_label].T,
                          composite_vec * product_other_factors)))

      if lim_cycle_detection_len > 1:
        limit_cycle_detectors[factor_label].UpdateBuffers(
            factor_states[factor_label], iter_idx)
        factor_has_limit_cycle.append(
            limit_cycle_detectors[factor_label].CheckForLimitCycle())
      else:
        factor_has_limit_cycle.append(False)

      factor_converged.append(
        (previous_states[factor_label] == factor_states[factor_label]).all())

    iter_idx += 1

    if live_plotting_obj is not None:
      live_plotting_obj.UpdatePlot(factor_states)

    if all(factor_converged):
      assert not all(factor_has_limit_cycle)
      converged = True

    if all(factor_has_limit_cycle):
      assert not all(factor_converged)
      limit_cycle_found = True
      cycle_lengths = {}
      for factor_label in factor_ordering:
        cycle_lengths[factor_label] = (
            limit_cycle_detectors[factor_label].LengthSmallestLimCycle())

  # The states are all +/- 1 so let's recast to int8
  for factor_label in factor_ordering:
    factor_states[factor_label] = factor_states[factor_label].astype('int8')

  # there exists a degeneracy where flipping the signs for any pair of factors
  # produces the same composite -- the dynamics can find a factorization which
  # which has a flipped sign compared to the ground truth for some of the
  # factors. This is easily detected and corrected by the following:
  for factor_label in factor_ordering:
    cosine_sims = cosine_sim(factor_states[factor_label],
                             factor_codebooks[factor_label])
    winner = np.argmax(np.abs(cosine_sims))  # nearest neighbor w.r.t cosine
    if cosine_sims[winner] < 0.0:
      factor_states[factor_label] = factor_states[factor_label] * -1

  if not silent:
    if converged:
      print('Converged in ', iter_idx, ' iterations')
    elif limit_cycle_found:
      print('Limit cycle detected at iteration ', iter_idx)
    else:
      print('Forcibly stopped at ', max_num_iters, ' iterations')

  if limit_cycle_found:
    lim_cycle_return = {'found': True, 'lengths': cycle_lengths}
  else:
    lim_cycle_return = {'found': False}

  return factor_states, iter_idx, lim_cycle_return


def activation(membrane_potential):
  """
  A *bipolar* version of the Heaviside function
  """
  temp = np.sign(membrane_potential)
  temp[temp == 0] = 1
  return temp


class LimitCycleCatcher(object):
  """
  This is a really simple datastructure for catching limit cycles

  We have to specify the maximum limit cycle length that we will look for
  but any time a limit cycle of this size or smaller is encountered, the
  CheckForLimitCycle method will return True, allowing us to take some action

  Parameters
  ----------
  state_space_size : int
      The number of components to our state space, the vector size
  max_lim_cycle_len : int, optional
      We will be sensitive to limit cycles up to this length. Default 20.
  """
  def __init__(self, state_space_size, max_lim_cycle_len=20):
    assert max_lim_cycle_len > 1, 'limit cycles can be of length 2 or larger'
    self.state_buffers = {}  # keeps track of the last k states
    self.buffer_repeat_counters = {}  # lim cycle found when repeats==k
    for k in range(2, max_lim_cycle_len+1):
      self.state_buffers[k] = np.zeros([k, state_space_size], dtype='float32')
      self.buffer_repeat_counters[k] = 0

  def UpdateBuffers(self, new_state, global_iter_idx):
    for buffer_sz in self.state_buffers:
      idx_in_buffer = global_iter_idx % buffer_sz
      if np.array_equal(self.state_buffers[buffer_sz][idx_in_buffer],
                        new_state):
        self.buffer_repeat_counters[buffer_sz] += 1
      else:
        self.state_buffers[buffer_sz][idx_in_buffer] = new_state
        self.buffer_repeat_counters[buffer_sz] = 0

  def CheckForLimitCycle(self):
    return any([self.buffer_repeat_counters[x] >= x
                for x in self.buffer_repeat_counters])

  def LengthSmallestLimCycle(self):
    if not any([self.buffer_repeat_counters[x] >= x
                for x in self.buffer_repeat_counters]):
      return 0
    else:
      return min([x for x in self.buffer_repeat_counters if
                  self.buffer_repeat_counters[x] >= x])
