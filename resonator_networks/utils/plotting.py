"""
Defines some useful utilities for plotting the evolution of a Resonator Network
"""
import copy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from utils.encoding_decoding import cosine_sim

class LiveResonatorPlot(object):
  """
  A container for a Matplotlib plot we'll use for live visualization

  Parameters
  ----------
  plot_type : str
      One of {'vec_img_viz', 'sim_bar_plots'}. Specifies which plot to
      show. The 'vec_img_viz' plots just display the target and current states
      as images. The 'sim_bar_plots' plot shows the similarity between each of
      the factors and the codebook as well as displays the similarity between
      the model's estimate of the composite vector and the composite vector
      itself.
  target_vectors : dictionary
      Contains the target vectors for each factor.
  factor_ordering : list
      Order to display the factors in. Just nice for consistent plotting
  query_vec : ndarray, optional
      The original composite vector given as a query to the Resonator Network.
      It will usually be the product of target vectors, but in general can
      have some additional noise corruption so we provide this as an additional
      input. Only necessary for the 'sim_bar_plots' type plot.
  codebooks: dict, optional.
      Keys label the factors and values are the corresponsing codebooks.
      Only necessary for the 'sim_bar_plots' type plot.
  image_size : (int, int), optional
      If applicable, dimensions of image visualization for vectors (in pixels).
      Only necessary for the 'vec_img_viz' type plot.
  """
  def __init__(self, plot_type, target_vectors, factor_ordering,
               query_vec=None, codebooks=None, image_size=None):
    assert plot_type in ['vec_img_viz', 'sim_bar_plots']
    self.vec_size = len(
        target_vectors[np.random.choice(list(target_vectors.keys()))])
    self.factor_ordering = factor_ordering
    self.plot_type = plot_type
    plt.ion()

    if plot_type == 'sim_bar_plots':
      assert codebooks is not None, 'please provide the codebooks as input'
      assert query_vec is not None, 'please provide the query vec as input'
      self.codebooks = copy.deepcopy(codebooks)
      self.query_vec = copy.deepcopy(query_vec)
      self.target_vectors = copy.deepcopy(target_vectors)
      self.barplot_refs = []  # hold references to the BarContainer objects

      # some constants to get GridSpec working for us
      mjm = 0.075  # {m}a{j}or {m}argin
      mnm = 1.0 # {m}i{n}or {m}argin
      vert_margin = 0.08
      horz_margin = 0.1

      gs_height = (((1.0 - 2*vert_margin) -
                   (mjm * (len(self.factor_ordering) - 1))) /
                   len(self.factor_ordering))

      fig = plt.figure(figsize=(15, 12))
      tab10colors = plt.get_cmap('tab10').colors

      for fac_idx in range(len(self.factor_ordering)):
        factor_label = self.factor_ordering[fac_idx]
        gs = GridSpec(6, 30)
        t = (1.0 - vert_margin) - fac_idx*gs_height - fac_idx*mjm
        gs.update(top=t, bottom=t - gs_height, left=horz_margin,
                  right=1.0-horz_margin, hspace=mnm, wspace=8*mnm)

        num_in_codebook = self.codebooks[factor_label].shape[1]

        # current states
        t_ax = plt.subplot(gs[:3, :18])
        self.barplot_refs.append(t_ax.bar(np.arange(num_in_codebook),
                                          np.zeros((num_in_codebook,)),
                                          color=tab10colors[fac_idx], width=1))
        t_ax.spines['top'].set_visible(False)
        t_ax.spines['right'].set_visible(False)
        t_ax.get_xaxis().set_ticks([])
        t_ax.get_yaxis().set_ticks([-1.0, 0.0, 1.0])
        t_ax.set_ylabel('Similarity', fontsize=12)
        t_ax.set_ylim(-1, 1)
        t_ax.yaxis.set_tick_params(labelsize=12)
        t_ax.text(0.02, 0.95, 'Current State', horizontalalignment='left',
                  verticalalignment='top', transform=t_ax.transAxes,
                  color='k', fontsize=14)
        if fac_idx == 0:
          t_ax.set_title('Current state of each factor', fontsize=18)

        # target similarity plot
        t_ax = plt.subplot(gs[3:, :18])
        target_csim = cosine_sim(
            target_vectors[factor_label],
            self.codebooks[factor_label])
        t_ax.bar(np.arange(num_in_codebook), target_csim,
                 color=tab10colors[fac_idx], width=1)
        t_ax.spines['top'].set_visible(False)
        t_ax.spines['right'].set_visible(False)
        t_ax.set_xlabel('Index in codebook', fontsize=12)
        t_ax.set_ylabel('Similarity', fontsize=12)
        t_ax.get_yaxis().set_ticks([-1.0, 0.0, 1.0])
        t_ax.text(0.02, 0.95, 'Target State', horizontalalignment='left',
                  verticalalignment='top', transform=t_ax.transAxes,
                  color='k', fontsize=14)
        if num_in_codebook > 10:
          t_ax.get_xaxis().set_ticks(
              np.arange(0, num_in_codebook, np.rint(num_in_codebook/10)))
        else:
          t_ax.get_xaxis().set_ticks(np.arange(num_in_codebook))
        t_ax.xaxis.set_tick_params(labelsize=12)
        t_ax.yaxis.set_tick_params(labelsize=12)

      # similarity between query composite and current estimated composite
      gs = GridSpec(3, 30)
      t_ax = plt.subplot(gs[1:2, 22:])
      # similarities to target
      self.lineplot_ref = Line2D([], [], color='k', linewidth=3)
      self.total_sim_saved = []

      t_ax.add_line(self.lineplot_ref)
      t_ax.set_ylim(-1.25, 1.25)
      t_ax.set_xlim(0, 20)  # we'll have to update the axis every ten steps
      t_ax.set_title(r'Similarity between $\mathbf{c}$ and $\hat{\mathbf{c}}$',
                     fontsize=18)
      t_ax.set_xlabel('Iteration number', fontsize=14)
      t_ax.set_ylabel('Cosine Similarity', fontsize=14)
      t_ax.xaxis.set_tick_params(labelsize=12)
      t_ax.yaxis.set_tick_params(labelsize=12)
      t_ax.yaxis.set_ticks([-1, 0, 1])
      t_ax.spines['top'].set_visible(False)
      t_ax.spines['right'].set_visible(False)
      self.sim_plot_ax_ref = t_ax


    if plot_type == 'vec_img_viz':
      if image_size is None:
        # we assume that vector is a square number and we will display
        # as a square image
        assert np.sqrt(self.vec_size) % 1 == 0
        self.image_size = (int(np.sqrt(self.vec_size)),
                           int(np.sqrt(self.vec_size)))
      else:
        self.image_size = image_size

      self.fig, self.axes = plt.subplots(
          len(factor_ordering), 2, figsize=(10, 15))
      self.fig.suptitle('Resonator State', fontsize='15')
      self.im_refs = []
      for fac_idx in range(len(self.factor_ordering)):
        factor_label = self.factor_ordering[fac_idx]
        self.im_refs.append([])
        maxval = np.max(target_vectors[factor_label])
        minval = np.min(target_vectors[factor_label])
        targ_im = self.axes[fac_idx][0].imshow(
            np.reshape(target_vectors[factor_label],
                       self.image_size), cmap='gray', vmin=minval, vmax=maxval)
        self.axes[fac_idx][0].set_title(
            'Target vector for ' + factor_label)
        self.axes[fac_idx][0].axis('off')
        self.im_refs[fac_idx].append(targ_im)

        res_im = self.axes[fac_idx][1].imshow(
            np.zeros(self.image_size), cmap='gray', vmin=minval, vmax=maxval)
        self.axes[fac_idx][1].set_title(
            'Current state for ' + factor_label)
        self.axes[fac_idx][1].axis('off')
        self.im_refs[fac_idx].append(res_im)

    plt.show(block=False)
    plt.draw()

  def UpdatePlot(self, current_state, wait_interval=0.001):
    if self.plot_type == 'sim_bar_plots':
      for f_idx in range(len(self.factor_ordering)):
        csim = cosine_sim(current_state[self.factor_ordering[f_idx]],
                          self.codebooks[self.factor_ordering[f_idx]])
        # really slow, should find a faster visualization solution
        for rect, ht in zip(self.barplot_refs[f_idx], csim):
          rect.set_height(ht)

      composite_est = np.product(np.array(
        [current_state[x] for x in current_state]), axis=0)
      self.total_sim_saved.append(cosine_sim(self.query_vec, composite_est))

      self.lineplot_ref.set_data(
          np.arange(len(self.total_sim_saved)), self.total_sim_saved)

      if len(self.total_sim_saved) % 20 == 0:
        self.sim_plot_ax_ref.set_xlim(0, len(self.total_sim_saved) + 20)
      if (len(self.total_sim_saved) > 1 and
          not np.isclose(self.total_sim_saved[-1], self.total_sim_saved[-2])):
        if self.total_sim_saved[-1] < self.total_sim_saved[-2]:
          self.lineplot_ref.set_color('r')
        else:
          self.lineplot_ref.set_color('k')
    else:
      for f_idx in range(len(self.factor_ordering)):
        self.im_refs[f_idx][-1].set_data(
            np.reshape(current_state[self.factor_ordering[f_idx]],
                       self.image_size))

    pause_without_refocus(wait_interval)
    #^ we could get a LOT faster plotting my using something other than
    #  plt.pause but this is quick an dirty...

  def ClosePlot(self):
    plt.close()
    # plus any other cleanup we may need


def pause_without_refocus(interval):
  backend = plt.rcParams['backend']
  if backend in matplotlib.rcsetup.interactive_bk:
    figManager = matplotlib._pylab_helpers.Gcf.get_active()
    if figManager is not None:
      canvas = figManager.canvas
      if canvas.figure.stale:
        canvas.draw()
      canvas.start_event_loop(interval)
      return
