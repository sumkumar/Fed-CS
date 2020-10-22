#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LISTA.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified : 2018-10-21

Implementation of Learned ISTA proposed by LeCun et al in 2010.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink
from models.LISTA_base import LISTA_base

class LISTA (LISTA_base):

    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, untied, coord, scope, name):
        """
        :A      : Numpy ndarray. Dictionary/Sensing matrix.
        :T      : Integer. Number of layers (depth) of this LISTA model.
        :lam    : Float. The initial weight of l1 loss term in LASSO.
        :untied : Boolean. Flag of whether weights are shared within layers.
        :scope  : String. Scope name of the model.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope
        self._name = name

        """ Set up layers."""
        self.setup_layers()


    # def setup_layers(self):
    #     """
    #     Implementation of LISTA model proposed by LeCun in 2010.

    #     :prob: Problem setting.
    #     :T: Number of layers in LISTA.
    #     :returns:
    #         :layers: List of tuples ( name, xh_, var_list )
    #             :name: description of layers.
    #             :xh: estimation of sparse code at current layer.
    #             :var_list: list of variables to be trained seperately.

    #     """
    #     Bs_     = []
    #     Ws_     = []
    #     thetas_ = []

    #     B = (np.transpose (self._A) / self._scale).astype (np.float32)
    #     W = np.eye (self._N, dtype=np.float32) - np.matmul (B, self._A)

    #     with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
    #         # constant
    #         self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

    #         Bs_.append (tf.get_variable (name=self._name+'B', dtype=tf.float32,
    #                                      initializer=B))
    #         Bs_ = Bs_ * self._T

    #         for t in range (self._T):
    #             thetas_.append (tf.get_variable (name=self._name+"theta_%d"%(t+1),
    #                                              dtype=tf.float32,
    #                                              initializer=self._theta))
    #             Ws_.append (tf.get_variable (name=self._name+"W_%d"%(t+1),
    #                                             dtype=tf.float32,
    #                                             initializer=W))

    #     # Collection of all trainable variables in the model layer by layer.
    #     # We name it as `vars_in_layer` because we will use it in the manner:
    #     # vars_in_layer [t]
    #     self.vars_in_layer = list (zip (Bs_, Ws_, thetas_))
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
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            Bs_.append (tf.get_variable (name=self._name+'B', dtype=tf.float32,
                                         initializer=B))
            Bs_ = Bs_ * self._T
            if not self._untied: # tied model
                Ws_.append (tf.get_variable (name=self._name+'W', dtype=tf.float32,
                                             initializer=W))
                Ws_ = Ws_ * self._T

            for t in range (self._T):
                thetas_.append (tf.get_variable (name=self._name+"theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name=self._name+"W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=W))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_, Ws_, thetas_))

    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_, W_, theta_ = self.vars_in_layer [t]
                # print(B_.shape)
                # print(W_.shape)
                # print(theta_.shape)
                By_ = tf.matmul (B_, y_)
                xh_ = shrink (tf.matmul (W_, xh_) + By_, theta_)
                xhs_.append (xh_)

        return xhs_

    # def set_weights(self,Avg_weights,name):
    #   with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
    #     # for t in range(self._T):
    #     #   B_,W_,theta_ = updated_weights[t]
    #     #   self.vars_in_layer[t] = (tf.get_variable(name="B_%d"%(t+1),dtype=tf.float32,initializer=B_),tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_),tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_))
    #       # self.vars_in_layer[t][1] = tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_)
    #       # self.vars_in_layer[t][2] = tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_)
              
    #     B_ = [tf.get_variable (name='B',dtype=tf.float32,
    #                       initializer=Avg_weights[0]),tf.get_variable (name='B',dtype=tf.float32,
    #                       initializer=Avg_weights[3])]
    #     W_ = [tf.get_variable (name='W_%d'%(1),dtype=tf.float32,
    #                       initializer=Avg_weights[1]),tf.get_variable (name='W_%d'%(2),dtype=tf.float32,
    #                       initializer=Avg_weights[4])]
    #     theta_ = [tf.get_variable (name='theta_%d'%(1),dtype=tf.float32,
    #                       initializer=Avg_weights[2]),tf.get_variable (name='theta_%d'%(2),dtype=tf.float32,
    #                       initializer=Avg_weights[5])]
        
    #     # print(sess.run(theta_))
    #     self.vars_in_layer = list(zip(B_,W_,theta_))  
    #     # print(sess.run(self.vars_in_layer[:][2])) 
    #   # print(len(B_))
    #   # print(B_[0].shape)
    #   # print(len(W_))
    #   # print(W_[0].shape)
    #   # print(len(theta_))
    #   # print(theta_[0].shape) 
    #   return self
    def set_weights(self,weights,sess):
      with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
        # for t in range(self._T):
        #   B_,W_,theta_ = updated_weights[t]
        #   self.vars_in_layer[t] = (tf.get_variable(name="B_%d"%(t+1),dtype=tf.float32,initializer=B_),tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_),tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_))
          # self.vars_in_layer[t][1] = tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_)
          # self.vars_in_layer[t][2] = tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_)
        # print(sess.run(Avg_weights[2]))
        """B_ = [tf.get_variable (name=name+'B',dtype=tf.float32,
                          initializer=Avg_weights[0]),tf.get_variable (name=name+'B',dtype=tf.float32,
                          initializer=Avg_weights[3])]
        W_ = [tf.get_variable (name=name+'W_%d'%(1),dtype=tf.float32,
                          initializer=Avg_weights[1]),tf.get_variable (name=name+'W_%d'%(2),dtype=tf.float32,
                          initializer=Avg_weights[4])]
        theta_ = [tf.get_variable (name=name+'theta_%d'%(1),dtype=tf.float32,
                          initializer=Avg_weights[2]),tf.get_variable (name=name+'theta_%d'%(2),dtype=tf.float32,
                          initializer=Avg_weights[5])]"""
        # assign_op = self.vars_in_layer[0][0].assign(Avg_weights[0])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][0].assign(Avg_weights[3])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[0][1].assign(Avg_weights[1])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][1].assign(Avg_weights[4])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[0][2].assign(Avg_weights[2])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][2].assign(Avg_weights[5])
        # sess.run(assign_op)

        layers = len(self.vars_in_layer)
        for i in range(layers):
            assign_op = self.vars_in_layer[i][0].assign(weights['B'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][1].assign(weights['W'][i])
            sess.run(assign_op)
            assign_op = self.vars_in_layer[i][2].assign(weights['theta'][i])
            sess.run(assign_op)
        # print(sess.run(theta_))
        #self.vars_in_layer = list(zip(B_,W_,theta_))  
        # print(sess.run(self.vars_in_layer[:][2])) 
      # print(len(B_))
      # print(B_[0].shape)
      # print(len(W_))
      # print(W_[0].shape)
      # print(len(theta_))
      # print(theta_[0].shape) 
      return self
    
    
    def set_weights_at_layer(self, weights, layer, sess):
      with tf.variable_scope (self._scope, reuse=tf.AUTO_REUSE) as vs:
        # for t in range(self._T):
        #   B_,W_,theta_ = updated_weights[t]
        #   self.vars_in_layer[t] = (tf.get_variable(name="B_%d"%(t+1),dtype=tf.float32,initializer=B_),tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_),tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_))
          # self.vars_in_layer[t][1] = tf.get_variable(name="W_%d"%(t+1),dtype=tf.float32,initializer=W_)
          # self.vars_in_layer[t][2] = tf.get_variable(name="theta_%d"%(t+1),dtype=tf.float32,initializer=theta_)
        # print(sess.run(Avg_weights[2]))
        """B_ = [tf.get_variable (name=name+'B',dtype=tf.float32,
                          initializer=Avg_weights[0]),tf.get_variable (name=name+'B',dtype=tf.float32,
                          initializer=Avg_weights[3])]
        W_ = [tf.get_variable (name=name+'W_%d'%(1),dtype=tf.float32,
                          initializer=Avg_weights[1]),tf.get_variable (name=name+'W_%d'%(2),dtype=tf.float32,
                          initializer=Avg_weights[4])]
        theta_ = [tf.get_variable (name=name+'theta_%d'%(1),dtype=tf.float32,
                          initializer=Avg_weights[2]),tf.get_variable (name=name+'theta_%d'%(2),dtype=tf.float32,
                          initializer=Avg_weights[5])]"""
        # assign_op = self.vars_in_layer[0][0].assign(Avg_weights[0])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][0].assign(Avg_weights[3])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[0][1].assign(Avg_weights[1])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][1].assign(Avg_weights[4])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[0][2].assign(Avg_weights[2])
        # sess.run(assign_op)
        # assign_op = self.vars_in_layer[1][2].assign(Avg_weights[5])
        # sess.run(assign_op)

        
        assign_op = self.vars_in_layer[layer][0].assign(weights['B'])
        sess.run(assign_op)
        assign_op = self.vars_in_layer[layer][1].assign(weights['W'])
        sess.run(assign_op)
        assign_op = self.vars_in_layer[layer][2].assign(weights['theta'])
        sess.run(assign_op)
        # print(sess.run(theta_))
        #self.vars_in_layer = list(zip(B_,W_,theta_))  
        # print(sess.run(self.vars_in_layer[:][2])) 
      # print(len(B_))
      # print(B_[0].shape)
      # print(len(W_))
      # print(W_[0].shape)
      # print(len(theta_))
      # print(theta_[0].shape) 
      return self
