import tensorflow as tf


def mask_generator_tf(p_m, x):
    """Generates a boolean mask for dataset

    Args:
        p_m (float): proportion of values to mask
        x (np.array): dataset to be masked

    Returns:
        mask (tf.Tensor): boolean mask with the same shape as x
    """
    mask = tf.keras.backend.random_bernoulli(x.shape, p_m)
    return mask


def pretext_generator_tf(m, x):
    """Generates a corrupted array

    Args:
        m (tf.Tensor): boolean mask with shape of x
        x (tf.Tensor): dataset to be corrupted

    Returns:
        m_new (tf.Tensor): updated boolean mask
        x_corrupt (tf.Tensor): corrupted dataset 
    """
    x_corrupt_features = []
    # Per column
    for i in range(x.shape[1]):
        # Reshuffle indices
        shuffled_index = tf.random.shuffle(range(x.shape[0]))
        x_b = tf.gather(x[:, i], shuffled_index)  # shuffled matrix
        x_corrupt_features.append(x[:, i] * (1 - m[:, i]) + m[:, i] * x_b)

    x_corrupt = tf.stack(x_corrupt_features, axis=1)
    m_new = tf.cast(x != x_corrupt, tf.int32)
    return m_new, x_corrupt


def to_vime_dataset(x, p_m, batch_size=1024, shuffle=False):
    # Generate mask
    m = mask_generator_tf(p_m, x)
    # Corrupt the dataset
    m, x_corr = pretext_generator_tf(m, tf.constant(x, dtype=tf.float32))
    
    autotune = tf.data.AUTOTUNE
    # TF Dataset with 3 inputs
    ds = tf.data.Dataset.from_tensor_slices((x_corr, {"mask": m, "feature": x})).batch(batch_size).prefetch(autotune)
    if shuffle:
        ds = ds.shuffle(len(m))
    
    # return ds for training and mask for evaluation
    return ds, m


def labelled_loss_fn(y, y_preds):
    bc = BinaryCrossentropy()
    return bc(y, y_preds)


def unlabelled_loss_fn(y):
    loss = tf.reduce_mean(tf.nn.moments(y, axes=1)[0])
    return loss


def semi_supervised_generator(X_l, X_u, y, bs=32):
    # Select the same number of random labelled and unlabelled examples
    batch_idx_u = np.random.choice(range(X_u.shape[0]), bs, replace=True)
    batch_idx_l = np.random.choice(range(X_l.shape[0]), bs, replace=True)
    
    # Output labelled X & y
    X_l_output = tf.constant(X_l[batch_idx_l, :], dtype=tf.float32)
    y_output = tf.constant(y[batch_idx_l], dtype=tf.float32)
    
    # Output unlabelled X
    X_u_output = tf.constant(X_u[batch_idx_u, :], dtype=tf.float32)

    yield X_l_output, y_output, X_u_output
