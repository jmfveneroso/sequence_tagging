import tensorflow as tf
import numpy as np

a = tf.constant([
  [
      [ 1 ],
      [ 2 ]
  ],
  [
      [ 3 ],
      [ 4 ]
  ]
])

b = tf.constant([
  [
      [ 5, 6 ],
      [ 5, 6 ]
  ],
  [
      [ 5, 6 ],
      [ 5, 6 ]
  ]
])

b = tf.constant([
  [ 3, 3 ],
  [ 3, 3 ]
])

# a = tf.constant([
#       [ 2, 2 ],
#       [ 2, 2 ]
# ])
# 
# b = tf.constant([
#       [ 1, 1 ],
#       [ 1, 1 ]
# ])

# c = tf.matmul(a, b)
print(a.shape)

# c = tf.tensordot(a, b, axes=1)
c = a * b
print(c)

with tf.Session() as sess:
  res = sess.run(c)
  print(res)
