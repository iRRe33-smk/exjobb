import tensorflow as tf
import numpy as np
nTech = 14
nParams = 7


S1 = tf.Variable(tf.random_normal_initializer((None, nTech)))
S2 = tf.Variable(tf.random_normal_initializer((None, nTech)))
u1 = tf.Variable(tf.random_normal_initializer((None, nTech)))
u2 = tf.Variable(tf.random_normal_initializer((None, nTech)))


C = tf.constant(np.random.randn(nTech, nParams))

score = tf.matmul((S1 + u1).T, C) - tf.matmul((S2 + u2).T, C)

grad_u1, grad_u2 = tf.gradients(score, [u1, u2])
hess_u1, hess_u2 = tf.hessians(score, [u1, u2])

# tensorflow är för vuxna. Sjukt jobbigt att skriva kod för
# Går säkert snabbare. Men tveksamt om v
