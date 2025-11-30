import tensorflow as tf

@tf.function
def augment_patch(patch):
    # Spatial flips
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)

    # Random 90-degree rotation
    k = tf.random.uniform(shape=[], minval=-3, maxval=3, dtype=tf.int32)
    patch = tf.image.rot90(patch, k)

    # angle = tf.random.uniform([], minval=-0.2 * 3.1416, maxval=0.2 * 3.1416)
    # patch = tfa.image.rotate(patch, angles=angle, interpolation='BILINEAR')

    return patch