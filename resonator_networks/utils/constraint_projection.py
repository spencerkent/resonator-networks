"""
Just a few useful methods for projection. Used by Projected Gradient Descent

For now we care about projection onto the simplex (with variable radius) and
the l1 ball (with variable radius).

The O(NlogN) simplex projection was first attributed to
.. [1] Held, M., Wolfe, P., Crowder, H. (1974). Validation of subgradient
       optimization. Mathematical Programming, 6, 62-88.

But it was more recently popularized and expanded on (drawing on several older
ideas) by
.. [2] Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008). Efficient
       projections onto the l1-ball for learning in high dimensions. ICML.
This paper presents a O(N) simplex projection based on a telescoping partition
rather than a full sort.

An important correction to [2] is given by [3] which reviews other efficient
methods of computing the simplex projection.
.. [3] Condat, L. (2016). Fast projection onto the simplex and the l1 ball.
       Mathematical Programming, Ser. A, 158:575-585.

My implementation is based on implementations by John Duchi and Adrien Gaidon
"""

import numpy as np

def project_simplex_nlogn(orig_vector, radius=1.0):
  """
  This is the simplest simplex projection method. It's O(nlogn), but works fine

  Parameters
  ----------
  orig_vector : ndarray(size=(n,))
      The vector to be projected onto the simplex
  radius : float, optional
      The radius of the simplex, \sum_i |orig_vector_i| = radius. Default 1.0.
  """
  assert radius > 0, "requested radius of the simplex must be positive"
  assert orig_vector.ndim == 1, "vector must be 1-d"
  # check if orig_vector is already on the simplex, then we're done
  if np.sum(orig_vector) == radius and np.alltrue(orig_vector >= 0):
    return orig_vector
  sorted_htl = np.sort(orig_vector)[::-1]  # sorted _h_igh _t_o _l_ow
  cumulative_sum = np.cumsum(sorted_htl)
  rho = np.nonzero(sorted_htl > ((cumulative_sum - radius) /
                                 np.arange(1, len(orig_vector) + 1)))[0][-1]
  #^ in terms of index-starting-at-zero convention
  theta = (cumulative_sum[rho] - radius) / (rho + 1)
  return (orig_vector - theta).clip(min=0)

def project_l1_ball(orig_vector, radius=1.0):
  """
  Computes the l1-ball projection from the simplex projection

  Parameters
  ----------
  orig_vector : ndarray (n,)
      The vector to be projected onto the simplex
  radius : float, optional
      The radius of the simplex, \sum_i |orig_vector_i| = radius. Default 1.0
  """
  assert radius > 0, "requested radius of the l1-ball must be positive"
  assert orig_vector.ndim == 1, "vector must be 1-d"
  abs_value_vec = np.abs(orig_vector)
  # check if this is already inside the ball, then we're done
  if np.sum(abs_value_vec) <= radius:
    return orig_vector
  simplex_proj = project_simplex_nlogn(abs_value_vec, radius)
  return np.sign(orig_vector) * simplex_proj

def proximal_l1(orig_vector, lambda_param):
  """
  Just the basic proximal operator for g() = \lambda||x||_1
  """
  return (np.sign(orig_vector) *
          np.maximum(np.abs(orig_vector) - lambda_param, 0.))

#TODO: add more efficient simplex projection. See [2] and correction in [3]
