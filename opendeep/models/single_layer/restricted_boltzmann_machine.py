"""
.. module:: restricted_boltzmann_machine

This module provides the RBM. http://deeplearning.net/tutorial/rbm.html

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.

Also see:
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
for optimizations
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import numpy
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils.cost import binary_crossentropy

log = logging.getLogger(__name__)

class RBM(Model):
    """
    This is a probabilistic, energy-based model.
    Basic binary implementation from:
    http://deeplearning.net/tutorial/rnnrbm.html
    """
    default = {
        'visible_activation': 'sigmoid',  # type of activation to use for visible activation
        'hidden_activation': 'sigmoid',  # type of activation to use for hidden activation
        'weights_init': 'uniform',  # either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0,  # mean for gaussian weights init
        'weights_std': 0.005,  # standard deviation for gaussian weights init
        'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
        'bias_init': 0.0,  # how to initialize the bias parameter
        'input_size': None,
        'hidden_size': None,
        'MRG': RNG_MRG.MRG_RandomStreams(1),  # default random number generator from Theano
        'rng': numpy.random.RandomState(1),  #default random number generator from Numpy
        'k': 15,  # the k steps used for CD-k or PCD-k with Gibbs sampling
        'outdir': 'outputs/rbm/'  # the output directory for this model's outputs
    }

    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, config=None, defaults=default,
                 input_size=None, hidden_size=None, visible_activation=None, hidden_activation=None,
                 weights_init=None, weights_mean=None, weights_std=None, weights_interval=None, bias_init=None,
                 MRG=None, rng=None, k=None, outdir=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.
        """
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        super(RBM, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # all configuration parameters are now in self!

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, hiddens_hook, or from parameters
        if self.inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            assert len(self.inputs_hook) == 2, 'Expected inputs_hook to be tuple!'  # make sure inputs_hook is a tuple
            self.input_size = self.inputs_hook[0] or self.input_size
            self.input = self.inputs_hook[1]
        else:
            # make the input a symbolic matrix
            self.input = T.fmatrix('V')

        # either grab the hidden's desired size from the parameter directly, or copy n_in
        self.hidden_size = self.hidden_size or self.input_size

        # deal with hiddens_hook
        if self.hiddens_hook is not None:
            # make sure hiddens_hook is a tuple
            assert len(self.hiddens_hook) == 2, 'Expected hiddens_hook to be tuple!'
            self.hidden_size = self.hiddens_hook[0] or self.hidden_size
            self.hiddens_init = self.hiddens_hook[1]
        else:
            self.hiddens_init = None

        # other specifications
        # visible activation function!
        self.visible_activation_func = get_activation_function(self.visible_activation)

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.visible_activation_func):
            self.visible_sampling = self.MRG.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary visible activation not supported yet!")
            raise NotImplementedError("Non-binary visible activation not supported yet!")

        # hidden activation function!
        self.hidden_activation_func = get_activation_function(self.hidden_activation)

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.hidden_activation_func):
            self.hidden_sampling = self.MRG.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary hidden activation not supported yet!")
            raise NotImplementedError("Non-binary hidden activation not supported yet!")

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            # make sure the params_hook has W (weights matrix) and bh, bv (bias vectors)
            assert len(self.params_hook) == 3, \
                "Expected 3 params (W, bh, bv) for RBM, found {0!s}!".format(len(self.params_hook))
            self.W, self.bh, self.bv = self.params_hook
            self.hidden_size = self.W.shape[1]
        else:
            self.W = get_weights(weights_init=self.weights_init,
                                 shape=(self.input_size, self.hidden_size),
                                 name="W",
                                 # if gaussian
                                 mean=self.weights_mean,
                                 std=self.weights_std,
                                 # if uniform
                                 interval=self.weights_interval)

            # grab the bias vectors
            self.bh = get_bias(shape=self.hidden_size, name="bh", init_values=self.bias_init)
            self.bv = get_bias(shape=self.input_size, name="bv", init_values=self.bias_init)

        # Finally have the three parameters
        self.params = [self.W, self.bh, self.bv]

        # Create the RBM graph!
        self.cost, self.monitors, self.updates, self.v_sample, self.h_sample = self.build_rbm()

        log.debug("Initialized an RBM shape %s",
                  str((self.input_size, self.hidden_size)))

    def build_rbm(self):
        """
        Creates the updates

        :return: The cost expression - free energy,
        monitor expression - binary cross-entropy to monitor training progress,
        updates dictionary - updates from the Gibbs sampling process,
        and last sample in the chain - last generated visible sample from the Gibbs process
        :rtype: List
        """
        # initialize from visibles if we aren't generating from some hiddens
        if self.hiddens_init is None:
            [_, v_chain, _, h_chain], updates = theano.scan(fn=lambda v: self.gibbs_step_vhv(v),
                                                            outputs_info=[None, self.input, None, None],
                                                            n_steps=self.k)
        # initialize from hiddens
        else:
            [_, v_chain, _, h_chain], updates = theano.scan(fn=lambda h: self.gibbs_step_hvh(h),
                                                            outputs_info=[None, None, None, self.hiddens_init],
                                                            n_steps=self.k)

        v_sample = v_chain[-1]
        h_sample = h_chain[-1]

        mean_v, _, _, _ = self.gibbs_step_vhv(v_sample)

        # some monitors
        # get rid of the -inf for the pseudo_log monitor (due to 0's and 1's in mean_v)
        # eps = 1e-8
        # zero_indices = T.eq(mean_v, 0.0).nonzero()
        # one_indices = T.eq(mean_v, 1.0).nonzero()
        # mean_v = T.inc_subtensor(x=mean_v[zero_indices], y=eps)
        # mean_v = T.inc_subtensor(x=mean_v[one_indices], y=-eps)
        pseudo_log = T.xlogx.xlogy0(self.input, mean_v) + T.xlogx.xlogy0(1 - self.input, 1 - mean_v)
        pseudo_log = pseudo_log.sum() / self.input.shape[0]
        crossentropy = T.mean(binary_crossentropy(mean_v, self.input))

        monitors = {'pseudo-log': pseudo_log, 'crossentropy': crossentropy}

        # the free-energy cost function!
        cost = (self.free_energy(self.input) - self.free_energy(v_sample)) / self.input.shape[0]

        return cost, monitors, updates, v_sample, h_sample

    @staticmethod
    def create_rbm(v, W, bv, bh, k, rng):
        '''
        Construct a k-step Gibbs chain starting at v for an RBM.

        v : Theano vector or matrix
            If a matrix, multiple chains will be run in parallel (batch).
        W : Theano matrix
            Weight matrix of the RBM.
        bv : Theano vector
            Visible bias vector of the RBM.
        bh : Theano vector
            Hidden bias vector of the RBM.
        k : scalar or Theano scalar
            Length of the Gibbs chain.

        Return a (v_sample, cost, monitor, updates) tuple:

        v_sample : Theano vector or matrix with the same shape as `v`
            Corresponds to the generated sample(s).
        cost : Theano scalar
            Expression whose gradient with respect to W, bv, bh is the CD-k
            approximation to the log-likelihood of `v` (training example) under the
            RBM. The cost is averaged in the batch case.
        monitor: Theano scalar
            Pseudo log-likelihood (also averaged in the batch case).
        updates: dictionary of Theano variable -> Theano variable
            The `updates` object returned by scan.
        '''
        def gibbs_step(v):
            mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
            h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                             dtype=theano.config.floatX)
            mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
            v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                             dtype=theano.config.floatX)
            return mean_v, v

        chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                     n_steps=k)
        v_sample = chain[-1]

        mean_v = gibbs_step(v_sample)[0]

        # some monitors
        # get rid of the -inf for the pseudo_log monitor (due to 0's and 1's in mean_v)
        # eps = 1e-8
        # zero_indices = T.eq(mean_v, 0.0).nonzero()
        # one_indices = T.eq(mean_v, 1.0).nonzero()
        # mean_v = T.inc_subtensor(x=mean_v[zero_indices], y=eps)
        # mean_v = T.inc_subtensor(x=mean_v[one_indices], y=-eps)
        pseudo_log = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        pseudo_log = pseudo_log.sum() / v.shape[0]

        crossentropy = T.mean(binary_crossentropy(mean_v, v))

        monitors = {'pseudo-log': pseudo_log, 'crossentropy': crossentropy}

        def free_energy(v):
            return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

        cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

        return v_sample, cost, monitors, updates

    def gibbs_step_vhv(self, v):
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.bh)
        h = self.hidden_sampling(size=mean_h.shape, n=1, p=mean_h,
                                 dtype=theano.config.floatX)
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.bv)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        return mean_v, v, mean_h, h

    def gibbs_step_hvh(self, h):
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.bv)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.bh)
        h = self.hidden_sampling(size=mean_h.shape, n=1, p=mean_h,
                                 dtype=theano.config.floatX)
        return mean_v, v, mean_h, h

    def free_energy(self, v):
        """
        The free-energy equation used for contrastive-divergence

        :param v: the theano tensor representing the visible layer
        :type v: theano tensor

        :return: the free energy
        :rtype: theano expression
        """
        vbias_term  = -(v * self.bv).sum()
        hidden_term = -T.log(1 + T.exp(T.dot(v, self.W) + self.bh)).sum()
        return vbias_term + hidden_term

    ####################
    # Model functions! #
    ####################
    def get_inputs(self):
        return self.input

    def get_hiddens(self):
        return self.h_sample

    def get_outputs(self):
        return self.v_sample

    def generate(self, initial=None):
        raise NotImplementedError()

    def get_train_cost(self):
        return self.cost

    def get_gradient(self, starting_gradient=None, cost=None, additional_cost=None):
        # consider v_sample constant when computing gradients
        # this actually keeps v_sample from being considered in the gradient, to set gradient to 0 instead,
        # use theano.gradient.zero_grad
        theano.gradient.disconnected_grad(self.v_sample)
        return super(RBM, self).get_gradient(starting_gradient, cost, additional_cost)

    def get_updates(self):
        return self.updates

    def get_monitors(self):
        return self.monitors

    def get_params(self):
        return self.params

    def save_args(self, args_file="rbm_config.pkl"):
        super(RBM, self).save_args(args_file)