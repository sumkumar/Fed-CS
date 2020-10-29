"""
Utility functions implemented in TensorFlow, including:
    - miscellaneous
    - shrinkage functions
"""

import tensorflow as tf


#######################   Misc   ###########################
def is_tensor(x):
    return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable))

####################   Shrinkage   #########################
def shrink(input_, theta_):
    """
    Soft thresholding function with input input_ and threshold theta_.
    """
    theta_ = tf.maximum( theta_, 0.0 )
    return tf.sign(input_) * tf.maximum( tf.abs(input_) - theta_, 0.0 )
