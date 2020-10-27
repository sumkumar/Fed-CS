"""
Implementation of the original Learned ISTA for real world image compressive
sensing experiments.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink
from models.LISTA_base import LISTA_base


class LISTA_cs (LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, Phi, D, T, lam, untied, coord, scope,name):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._Phi  = Phi.astype (np.float32)
        self._D    = D.astype (np.float32)
        self._A    = np.matmul (self._Phi, self._D)
        self._T    = T
        self._lam  = lam
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]
        self._N    = self._D.shape [1]

        self._scale = 1.001 * np.linalg.norm (self._A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope
        self._name  = name
        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Bs_     = []
        Ws_     = []
        thetas_ = []

        B = (np.transpose (self._A) / self._scale).astype (np.float32)
        W = np.eye (self._N, dtype=np.float32) - np.matmul (B, self._A)

        with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._kD_   = tf.constant (value=self._D, dtype=tf.float32)
            self._kA_   = tf.constant (value=self._A, dtype=tf.float32)

            # variables
            self._vD_   = tf.get_variable (name=self._name + 'D', dtype=tf.float32,
                                           initializer=self._D)
            Bs_.append (tf.get_variable (name=self._name + 'B', dtype=tf.float32,
                                         initializer=B))
            Bs_ = Bs_ * self._T
            if not self._untied: # tied model
                Ws_.append (tf.get_variable (name=self._name + 'W', dtype=tf.float32,
                                             initializer=W))
                Ws_ = Ws_ * self._T

            for t in range (self._T):
                thetas_.append (tf.get_variable (name=self._name + "theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name=self._name + "W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=W))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        # Note here the last element of `self.vars_in_layer` is
        # (W_, theta_, vD_)
        self.vars_in_layer = list (zip (Bs_ [:-1], Ws_ [:-1], thetas_ [:-1]))
        self.vars_in_layer.append ((Bs_ [-1], Ws_ [-1], thetas_ [-1], self._vD_, ))


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        fhs_  = [] # collection of the regressed signals

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)
        fhs_.append (tf.matmul (self._kD_, xh_))

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                if t < self._T - 1:
                    B_, W_, theta_ = self.vars_in_layer [t]
                    D_ = self._kD_
                else:
                    B_, W_, theta_, D_ = self.vars_in_layer [t]

                By_ = tf.matmul (B_, y_)
                xh_ = shrink (tf.matmul (W_, xh_) + By_, theta_)
                xhs_.append (xh_)

                fhs_.append (tf.matmul (D_, xh_))

        return xhs_, fhs_

    def set_weights(self,weights,sess):
      with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
        
        layers = len(self.vars_in_layer)
        for i in range(layers):
          if i == layers-1:
            assign_op = self.vars_in_layer[i][0].assign(weights['B'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][1].assign(weights['W'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][2].assign(weights['theta'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][3].assign(weights['D'])
            sess.run(assign_op)
          else:  
            assign_op = self.vars_in_layer[i][0].assign(weights['B'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][1].assign(weights['W'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][2].assign(weights['theta'][i])
            sess.run(assign_op)
 
      return self
    
    def set_weights_at_layer(self, weights, layer, sess):
      with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:

        if layer == self._T-1:
          assign_op = self.vars_in_layer[layer][0].assign(weights['B'])
          sess.run(assign_op)
          assign_op = self.vars_in_layer[layer][1].assign(weights['W'])
          sess.run(assign_op)
          assign_op = self.vars_in_layer[layer][2].assign(weights['theta'])
          sess.run(assign_op)
          assign_op = self.vars_in_layer[layer][3].assign(weights['D'])
          sess.run(assign_op)
        else:
          assign_op = self.vars_in_layer[layer][0].assign(weights['B'])
          sess.run(assign_op)
          assign_op = self.vars_in_layer[layer][1].assign(weights['W'])
          sess.run(assign_op)
          assign_op = self.vars_in_layer[layer][2].assign(weights['theta'])
          sess.run(assign_op)

      return self