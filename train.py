import numpy as np
import tensorflow as tf


def lstm_cell_with_dropout(state_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        state_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
    return cell


def blstm_layer_with_dropout(inputs, seqlen, state_size, keep_prob, scope):
    fw_cell = lstm_cell_with_dropout(state_size, keep_prob)
    bw_cell = lstm_cell_with_dropout(state_size, keep_prob)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        sequence_length=seqlen,
        dtype=tf.float32,
        scope=scope)
    return tf.concat([output_fw, output_bw], axis=-1)


def blstm_model(embedding_vectors, num_layers, state_size, n_classes, keep_prob=0.5, tune_embeddings=False):
    x = tf.placeholder(tf.int32, [None, None], name='x')
    seqlen = tf.placeholder(tf.int32, [None], name='seqlen')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

    max_length = tf.shape(x)[1]

    W = tf.Variable(
        embedding_vectors,
        trainable=tune_embeddings,
        name="word_embeddings")
    rnn_inputs = tf.nn.embedding_lookup(W, x)

    for i in range(num_layers):
        with tf.name_scope("BLSTM-{}".format(i)) as scope:
            rnn_inputs = blstm_layer_with_dropout(
                rnn_inputs, seqlen, state_size, keep_prob, scope)

    with tf.name_scope('dense_linear'):
        logit_inputs = tf.reshape(rnn_inputs, [-1, 2 * state_size])
        logits = tf.layers.dense(logit_inputs, n_classes)
        logits = tf.reshape(logits, [-1, max_length, n_classes])

    with tf.name_scope('loss'):
        seqlen_mask = tf.sequence_mask(
            seqlen, maxlen=max_length, name='sequence_mask')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name="cross_entropy")
        loss = tf.boolean_mask(loss, seqlen_mask)
        loss = tf.reduce_mean(loss, name="mean_loss")

    return {
        'inputs': {
            'x': x,
            'seqlen': seqlen
        },
        'labels': labels,
        'loss': loss
    }


if __name__ == '__main__':
    logdir = 'log'
    file_writer = tf.summary.FileWriter(logdir)

    n_classes = 10
    num_layers = 2
    state_size = 25
    vocab_size = 10
    embedding_size = 150
    embedding_vectors = np.random.rand(
        vocab_size, embedding_size).astype(np.float32)
    model = lstm_model(embedding_vectors, num_layers, state_size, n_classes)
    with tf.Session() as sess:
        file_writer.add_graph(sess.graph)

    file_writer.close()
