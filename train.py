#coding=utf-8
import tensorflow as tf
import time
import os

from data_processor import TextLoader
from model import Model

flags = tf.app.flags

flags.DEFINE_string("data_dir", "data/tinyshakespeare", "Data directory")
flags.DEFINE_string("save_dir", "save", "Directory to store checkpoints")
flags.DEFINE_string("log_dir", "log", "Directory to store tensorboard")
flags.DEFINE_integer("save_every", 100, "Save frequency. Number of passes between checkpoints of the model.")
flags.DEFINE_string("model", "lstm", "Choice: lstm, rnn, gru, nas")
flags.DEFINE_integer("rnn_size", 128, "Size of RNN hidden state")
flags.DEFINE_integer("num_layers", 2, "Number of layers in RNN")
flags.DEFINE_integer("seq_length", 50, "RNN sequence length")
flags.DEFINE_integer("batch_size", 50, "Batch size, Number of sequences propagated through the network in parallel.")
flags.DEFINE_integer("num_epochs", 50, "Number of full pass through training")
flags.DEFINE_float("grad_clip", 5.0, "Clip gradients at this value")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
flags.DEFINE_float("decay_rate", 0.97, "Decay rate")
flags.DEFINE_float("output_keep_prob", 1.0, "Probability of keeping weights in the hidden layer")
flags.DEFINE_float("input_keep_prob", 1.0, "Probability of keeping weights in the input layer")
FLAGS = flags.FLAGS

RNNOPTIONS = ["rnn", "lstm", "gru", "nas"]

def main(unusedargv):
    if FLAGS.model not in RNNOPTIONS:
        raise ValueError("Invalid params. Optional is rnn, lstm, gru, nas")
    data_loader = TextLoader(FLAGS.data_dir, FLAGS.batch_size, FLAGS.seq_length)
    FLAGS.vocab_size = data_loader.vocab_size


    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    if not os.path.isdir(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,        log_device_placement=False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            model = Model(FLAGS)

            global_step = tf.Variable(0, name = 'global_step', trainable = False)

            grads_and_vars = zip(model.grads, model.tvars)

            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # summary
            logits_summary = tf.summary.histogram('logits', model.logits)
            loss_summary = tf.summary.histogram('loss', model.loss)
            trainloss_summary = tf.summary.scalar('train_loss', model.cost)

            train_summary_op = tf.summary.merge([logits_summary, loss_summary, trainloss_summary, grad_summaries_merged])

            writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            def train_step(x_batch, y_batch, epoch, state, b):
                start = time.time()
                feed = {
                    model.input_data: x,
                    model.targets: y
                }
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                summ, train_loss, state, _ = sess.run([train_summary_op, model.cost, model.final_state, train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(epoch * data_loader.num_batches + b,
                              FLAGS.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

            for e in range(FLAGS.num_epochs):
                sess.run(tf.assign(model.lr,
                               FLAGS.learning_rate * (FLAGS.decay_rate ** e)))
                data_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)
                for b in range(data_loader.num_batches):
                    x, y = data_loader.next_batch()
                    train_step(x, y, e, state, b)
                if (e * data_loader.num_batches + b) % FLAGS.save_every == 0\
                        or (e == FLAGS.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(FAGS.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == "__main__":
    tf.app.run()
