"""
NumPy implementation of benchmark algorithms

Benchmarks implemented in this file:
* Alternating Least Squares
* Iterative Soft Thresholding (ISTA)
* Fast Iterative Soft Thresholding (FISTA)
* Projected Gradient Descent
* Multiplicative Weights
* Map Seeking Circuits
"""
import copy
import numpy as np

from utils.encoding_decoding import cosine_sim
from utils.constraint_projection import project_simplex_nlogn
from utils.constraint_projection import proximal_l1

def run(composite_vec, factor_codebooks, algorithm, algorithm_params,
        coeff_init=None, max_num_iters=10000,
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
  algorithm : str
      One of {'Alternating Least Squares',
              'Iterative Soft Thresholding',
              'Fast Iterative Soft Thresholding',
              'Projected Gradient Descent',
              'Multiplicative Weights',
              'Map Seeking Circuits'}
  algorithm_params : dictionary
      Depends on the algorithm, but in general will specify the following:
      'convergence_tol' : float
      'stepsize' : float (for all but ALS)
      'sparsity_weight' : float (for ISTA and FISTA)
      'clipping_threshold' : float (for MSC)
  coeff_init : dictionary, optional
      We may want to manually specify the initial state of each factor, in
      which case we can do that here. Whereas with Resonator Networks one
      initializes the post-theshold state \hat{\mathbf{x}}_f, here we
      initialize the superposition coefficients \mathbf{a}_f (which is what
      we update in each iteration and determines the resulting state).
      Each value is a 1d array giving the initial setting for the
      coefficients of each factor. Default None.
  max_num_iters : int, optional
      We can optionally stop the algorithms after this many discrete-time
      iterations. Probably unnecessary for the benchmarks, as they all converge
      on their own. Default 10000.
  live_plotting_obj : ResonatorLivePlot, optional
      Used to display updates to the algorithm in a Matplotlib window.
      This is the MOST BASIC way to do this and probably won't run any
      faster than about 30 frames a second. Other solutions exit.
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
      Number of iterations taken by dynamics until either convergence or
      max_num_iters reached.
  """
  # Boilerplate parameter checking
  assert composite_vec.dtype == 'int8'
  assert algorithm in [
      'Alternating Least Squares', 'Iterative Soft Thresholding',
      'Fast Iterative Soft Thresholding', 'Projected Gradient Descent',
      'Multiplicative Weights', 'Map Seeking Circuits']
  assert 'convergence_tol' in algorithm_params
  if algorithm != 'Alternating Least Squares':
    if algorithm not in ['Iterative Soft Thresholding',
                         'Fast Iterative Soft Thresholding']:
      assert 'stepsize' in algorithm_params
      if algorithm == 'Multiplicative Weights':
        assert algorithm_params['stepsize'] <= 0.5
    if algorithm in ['Iterative Soft Thresholding',
                     'Fast Iterative Soft Thresholding']:
      assert 'sparsity_weight' in algorithm_params
    if algorithm == 'Map Seeking Circuits':
      assert 'clipping_threshold' in algorithm_params
  if coeff_init is not None:
    assert all([x in factor_codebooks for x in coeff_init])

  # Initialization
  factor_states = dict.fromkeys(factor_codebooks)
  factor_states_coeffs = dict.fromkeys(factor_codebooks)  # a_f
  factor_ordering = list(factor_codebooks.keys())  # update in consistent order
  if algorithm == 'Multiplicative Weights':
    # the weights are a set of auxiliary variables which determine the coeffs
    factor_m_weights = dict.fromkeys(factor_codebooks)
  for factor_label in factor_ordering:
    assert factor_codebooks[factor_label].dtype == 'int8'
    num_codevecs = factor_codebooks[factor_label].shape[1]
    if coeff_init is not None:
      if algorithm != 'Multiplicative Weights':
        factor_states_coeffs[factor_label] = np.copy(
            coeff_init[factor_label]).astype('float32')
      else:
        # we initialize the weights, which set the coefficients
        factor_m_weights[factor_label] = np.copy(
            coeff_init[factor_label]).astype('float64')
        factor_states_coeffs[factor_label] = (factor_m_weights[factor_label] /
            np.sum(factor_m_weights[factor_label]))
    else:
      # each alg has a slightly different initialization
      if algorithm in ['Projected Gradient Descent', 'Multiplicative Weights']:
        if algorithm == 'Multiplicative Weights':
          factor_m_weights[factor_label] = np.ones(
              num_codevecs, dtype='float64')
        factor_states_coeffs[factor_label] = (
            np.ones(num_codevecs, dtype='float32') / num_codevecs)
      else:
        factor_states_coeffs[factor_label] = np.ones(
            num_codevecs, dtype='float32')
        # ^one can rescale this to be on the simplex or l1 ball but it does
        #  not appear to make a significant difference

    factor_states[factor_label] = np.dot(
        factor_codebooks[factor_label], factor_states_coeffs[factor_label])

  if algorithm == 'Fast Iterative Soft Thresholding':
    # auxiliary points used to accelerate the descent, one for each factor
    alpha_t = {x: 1.0 for x in factor_ordering}
    alpha_tminusone = {x: 1.0 for x in factor_ordering}
    aux_pts_coeffs = {x: np.copy(factor_states_coeffs[x])
                      for x in factor_ordering}  # this is p_f[t+1] in paper
    aux_pts = {x: np.dot(factor_codebooks[x], aux_pts_coeffs[x])
               for x in factor_ordering}

  if live_plotting_obj is not None:
    live_plotting_obj.UpdatePlot(factor_states)

  iter_idx = 0
  converged = False
  while not converged and iter_idx < max_num_iters:
    previous_coeffs = copy.deepcopy(factor_states_coeffs)
    factor_converged = []  # convergence of each factor individually
    for factor_label in factor_ordering:
      product_other_factors = np.product(np.array([factor_states[x]
        for x in factor_states if x != factor_label]), axis=0)

      if algorithm == 'Alternating Least Squares':
        xi = product_other_factors[:, None] * factor_codebooks[factor_label]
        factor_states_coeffs[factor_label] = np.dot(np.linalg.pinv(xi),
                                                    composite_vec)

      # The other algs need to compute the gradient
      if algorithm in ['Projected Gradient Descent',
          'Multiplicative Weights', 'Map Seeking Circuits']:
        # our reference versions of these algs use the inner product loss fn
        # MW also seems to work with the squared error loss
        gradient = -1. * np.dot(factor_codebooks[factor_label].T,
            composite_vec * product_other_factors)

        if algorithm == 'Projected Gradient Descent':
          factor_states_coeffs[factor_label] = project_simplex_nlogn(
              factor_states_coeffs[factor_label] -
              algorithm_params['stepsize'] * gradient)

        elif algorithm == 'Multiplicative Weights':
          normalized_gradient = gradient / np.max(np.abs(gradient))
          with np.errstate(over='raise'):
            try:
              factor_m_weights[factor_label] = (
                  factor_m_weights[factor_label] *
                  (1. - algorithm_params['stepsize'] * normalized_gradient))
            except:
              print('We caught an overflow exception in the weight updates')
              # Happens rarely. Somewhat alleviated by using float64 for
              # weights instead of float32. If desired, can ignore by
              # checking num_iters return for null.
              return factor_states, None
          factor_states_coeffs[factor_label] = (factor_m_weights[factor_label]
              / np.sum(factor_m_weights[factor_label]))

        else:
          # Map Seeking Circuits
          if np.min(gradient) != 0:
            upd_direction = 1 + gradient / np.abs(np.min(gradient))
          else:
            # avoid divide by zero on the update. Don't update anything,
            # wait for the other factors to update
            upd_direction = np.zeros(len(factor_states_coeffs[factor_label]))
          factor_states_coeffs[factor_label] = (
              factor_states_coeffs[factor_label] -
              algorithm_params['stepsize'] * upd_direction)
          thresh_inds = (factor_states_coeffs[factor_label] <
                         algorithm_params['clipping_threshold'])
          factor_states_coeffs[factor_label][thresh_inds] = 0.

      elif algorithm in ['Iterative Soft Thresholding',
                         'Fast Iterative Soft Thresholding']:
        # the dynamic stepsize is set by 1/L where L is the Lipschitz constant
        # of the gradient. This changes in each iteration, because of the
        # change in the other factors. It can be the most expensive part of
        # of these particular benchmarks.
        l_const = np.real(np.linalg.eigh(np.dot(
          factor_codebooks[factor_label].T,
          np.dot(np.diag(np.square(product_other_factors)),
                 factor_codebooks[factor_label])))[0][-1])
        # ^if any of the other factors is all-zeros (could happen if sparsity
        #  weight is set too large) then this will be zero, which means the
        #  stepsize is inifinitely large. Assuming this is just a transient
        #  thing, let's set the stepsize to be small but nonzero for this update
        if l_const != 0:
          stepsize = 1. / l_const
        else:
          stepsize = 1. / 1000

        if algorithm == 'Iterative Soft Thresholding':
          # this alg uses the gradient of the squared error loss fn
          gradient = -1. * np.dot(factor_codebooks[factor_label].T,
              (composite_vec -
                factor_states[factor_label] * product_other_factors)
              * product_other_factors)
          factor_states_coeffs[factor_label] = proximal_l1(
              factor_states_coeffs[factor_label] - stepsize * gradient,
              algorithm_params['sparsity_weight'] * stepsize)

        elif algorithm == 'Fast Iterative Soft Thresholding':
          # this alg uses the gradient of the squared error loss fn
          # *evaluated at the auxiliary coefficients, p_f[t+1]*
          gradient = -1. * np.dot(factor_codebooks[factor_label].T,
              (composite_vec - aux_pts[factor_label] * product_other_factors)
              * product_other_factors)
          factor_states_coeffs[factor_label] = proximal_l1(
              aux_pts_coeffs[factor_label] - stepsize * gradient,
              algorithm_params['sparsity_weight'] * stepsize)

          alpha_tminusone[factor_label] = alpha_t[factor_label]
          alpha_t[factor_label] = (
              1+(1+(4*alpha_tminusone[factor_label]**2))**0.5)/2
          beta_t = (alpha_tminusone[factor_label] - 1) / alpha_t[factor_label]
          aux_pts_coeffs[factor_label] = (factor_states_coeffs[factor_label] +
              beta_t * (factor_states_coeffs[factor_label] -
                               previous_coeffs[factor_label]))
          aux_pts[factor_label] = np.dot(factor_codebooks[factor_label],
                                         aux_pts_coeffs[factor_label])

      # Update "states" based on coefficients.
      factor_states[factor_label] = np.dot(
          factor_codebooks[factor_label], factor_states_coeffs[factor_label])
      # ^you can and should experiment with putting different activation
      #  functions on this. With one exception, we found none that
      #  significantly improved performance. ALS is often formulated with a
      #  scalar renormalization of the state in each iteration, for instance.
      #  The one important exception is putting the sign function on ALS, which
      #  we show in the paper gives a Resonator Network with OLS weights.
      #  Beware that depending on the details, adding an activation function
      #  here may affect convergence guarantees.

      # check convergence of the coefficients
      if algorithm == 'Multiplicative Weights':
        # MW adapts the coefficients very slowly at first so a small change
        # isn't a sufficient condition for convergence. At convergence, however,
        # most of the coefficients will be close to zero.
        num_coeff_activated = np.sum(factor_states_coeffs[factor_label] > 1e-2)
        if num_coeff_activated / len(factor_states_coeffs[factor_label]) > 0.5:
          # early on in training, ignore criterion
          factor_converged.append(False)
        else:
          factor_converged.append((np.mean(np.abs(
            (previous_coeffs[factor_label] -
              factor_states_coeffs[factor_label]))) /
            algorithm_params['stepsize']) < algorithm_params['convergence_tol'])
      elif algorithm in ['Iterative Soft Thresholding',
                         'Fast Iterative Soft Thresholding']:
        # since the stepsize is changing at each iteration and can get very
        # small, I found it more sensible to NOT normalize by the stepsize
        # in this case. A convergence tol in the interval
        # [10^(-4), 10^(-6)] is reasonable.
        factor_converged.append(np.mean(np.abs(
          previous_coeffs[factor_label] - factor_states_coeffs[factor_label]))
          < algorithm_params['convergence_tol'])
      else:
        # Everything else
        if algorithm == 'Alternating Least Squares':
          delta_divisor = 1.0  # there's no stepsize in ALS
        else:
          delta_divisor = algorithm_params['stepsize']
        factor_converged.append((np.mean(np.abs(
          previous_coeffs[factor_label] - factor_states_coeffs[factor_label]))
          / delta_divisor) < algorithm_params['convergence_tol'])

    iter_idx += 1

    if live_plotting_obj is not None:
      live_plotting_obj.UpdatePlot(factor_states)

    if all(factor_converged):
      converged = True

  # we apply the same degeneracy disambiguation as we do for Resonator Networks
  # (See the implementation in rn_numpy.py. This is really only needed for
  # ISTA, FISTA, and ALS, but we do it across the board for consistency
  for factor_label in factor_ordering:
    cosine_sims = cosine_sim(factor_states[factor_label],
                             factor_codebooks[factor_label])
    winner = np.argmax(np.abs(cosine_sims))  # nearest neighbor w.r.t cosine
    if cosine_sims[winner] < 0.0:
      factor_states[factor_label] = factor_states[factor_label] * -1

  if not silent:
    if converged:
      print('Converged in ', iter_idx, ' iterations')
    else:
      print('Forcibly stopped at ', max_num_iters, ' iterations')

  return factor_states, iter_idx
