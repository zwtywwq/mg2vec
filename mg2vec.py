import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import random
import math

# parser = argparse.ArgumentParser(description="mg2vec skip-gram version")
# parser.add_argument("--alpha", required=True, help='first and second order ratio')
# parser.add_argument("--dim", required=True, help='dimension')
# parser.add_argument("--dataset", required=True, help='dataset')
# args = parser.parse_args()

data_index = 0
epoch_to_train = 3
print_index = 3
epoch_index = 1
batch_size = 512
embedding_size = 128
num_sampled = 10
# init_width = 0.5 / embedding_size
alpha = 0.5
dataset_name = 'dblp'
print(dataset_name)


# generate train file for unsupervised loss
def read_train_file(filename_r):
    node_index = 0
    mg_index = 0
    instance_count = 0
    node_dict = dict()
    mg_dict = dict()
    mg_num_dict = dict()
    unigram = list()
    train_file = list()
    with open(filename_r, 'r') as f:
        for line in f:
            temp = list(line.strip('\n').split())
            if temp[2][1:] not in mg_dict:
                mg_dict[temp[2][1:]] = mg_index
                mg_num_dict[temp[2][1:]] = float(temp[3][1:])
                mg_index += 1
            else:
                mg_num_dict[temp[2][1:]] += float(temp[3][1:])
            if temp[0] not in node_dict:
                node_dict[temp[0]] = node_index
                node_index += 1
            if temp[1] not in node_dict:
                node_dict[temp[1]] = node_index
                node_index += 1
    node_reverse_dict = dict(zip(node_dict.values(), node_dict.keys()))
    mg_reverse_dict = dict(zip(mg_dict.values(), mg_dict.keys()))
    for i in range(mg_index):
        unigram.append(int(math.ceil(mg_num_dict[mg_reverse_dict[i]])))
    with open(filename_r, 'r') as f:
        for line in f:
            temp = list(line.strip('\n').split())
            temp[0] = node_dict[temp[0]]
            temp[1] = node_dict[temp[1]]
            temp[2] = mg_dict[temp[2][1:]]
            train_file.append(temp)
            instance_count += 1
    random.shuffle(train_file)
    return node_index, mg_index, instance_count, node_reverse_dict, unigram, train_file, node_dict, mg_reverse_dict


node_num, mg_num, instance_num, index2node, unigram_table, file4train, node2index, index2mg = read_train_file(
    './dataset/sample_metagraph_stats')

print('node num is %d' % node_num)
print('mg num is %d' % mg_num)
print('instance num is %d' % instance_num)


def generate_batch(dataset):
    global data_index
    global epoch_index
    length = len(dataset)

    batch_a = np.ndarray(shape=(batch_size,), dtype=np.int32)
    batch_b = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,), dtype=np.int32)
    freqs = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    o2weight = np.ndarray(shape=(batch_size, 1), dtype=np.float32)

    for i in range(batch_size):
        if data_index == 0:
            print('epoch %d start' % epoch_index)
            epoch_index += 1
        batch_a[i] = int(dataset[data_index][0])
        batch_b[i] = int(dataset[data_index][1])
        if batch_a[i] == batch_b[i]:
            o2weight[i] = 1 - alpha
        else:
            o2weight[i] = alpha
        labels[i] = int(dataset[data_index][2])
        freqs[i] = float(dataset[data_index][3][1:])
        data_index = data_index + 1
        if data_index >= length:
            data_index = 0
            random.shuffle(file4train)
    return batch_a, batch_b, labels, freqs, o2weight


def unsupervised_loss_func():
    m_embed_pos = tf.nn.embedding_lookup(m_embeddings, train_labels)
    labels = tf.reshape(
        tf.cast(train_labels, dtype=tf.int64), [batch_size, 1])
    sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=mg_num,
        unigrams=unigram_table
    )
    m_embed_sampled = tf.nn.embedding_lookup(m_embeddings, sampled_ids)
    n_embed = tf.add(tf.multiply(tf.nn.relu(tf.matmul(n_embed_con, n_w_t) + n_b), inputs_pa),
                     tf.multiply(n_embed_a, inputs_pb))
    # n_embed = tf.nn.relu(tf.matmul(n_embed_con, n_w_t) + n_b)
    true_logits = tf.reduce_sum(tf.multiply(n_embed, m_embed_pos), axis=1, keepdims=True)
    sampled_logits = tf.matmul(n_embed, m_embed_sampled, transpose_b=True)
    logits = tf.concat([true_logits, sampled_logits], axis=1)
    labels = tf.concat([tf.ones_like(true_logits), tf.zeros_like(sampled_logits)], axis=1)
    xent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                         axis=1, keepdims=True)
    unsupervised_loss = tf.reduce_mean(tf.multiply(train_weight, tf.multiply(train_freqs, xent)))
    return unsupervised_loss


graph = tf.Graph()

with graph.as_default():
    # Inputs
    train_inputs_a = tf.placeholder(tf.int32)
    train_inputs_b = tf.placeholder(tf.int32)
    train_labels = tf.placeholder(tf.int32)
    train_freqs = tf.placeholder(tf.float32)
    train_weight = tf.placeholder(tf.float32)

    n_embeddings = tf.get_variable("n_embeddings", shape=[node_num, embedding_size],
                                   initializer=tf.contrib.layers.variance_scaling_initializer())
    m_embeddings = tf.get_variable("m_embeddings", shape=[mg_num, embedding_size],
                                   initializer=tf.contrib.layers.variance_scaling_initializer())

    n_w_t = tf.get_variable("n_w_t", shape=[embedding_size * 2, embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer())
    n_b = tf.Variable(tf.zeros([embedding_size]))

    n_embed_a = tf.nn.embedding_lookup(n_embeddings, train_inputs_a)
    n_embed_b = tf.nn.embedding_lookup(n_embeddings, train_inputs_b)
    inputs_pa = tf.reshape(tf.cast(tf.logical_not(tf.equal(train_inputs_a, train_inputs_b)), dtype=tf.float32),
                           [batch_size, 1])
    inputs_pb = tf.reshape(tf.cast(tf.equal(train_inputs_a, train_inputs_b), dtype=tf.float32), [batch_size, 1])
    n_embed_con = tf.concat([n_embed_a, n_embed_b], axis=1)

    loss_0 = unsupervised_loss_func()
    loss = loss_0

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999, 1e-08).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(n_embeddings), 1, keepdims=True))
    normalized_embeddings = n_embeddings / norm

    m_norm = tf.sqrt(tf.reduce_sum(tf.square(m_embeddings), 1, keepdims=True))
    normalized_m_embeddings = m_embeddings / m_norm

    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config) as session:
    writer = tf.summary.FileWriter('log', session.graph)
    init.run()
    print('Initialized')
    average_loss = 0
    step = 0
    while epoch_index <= epoch_to_train + 1:
        step += 1
        batch_inputs_a, batch_inputs_b, batch_labels, freqs, o2weight = generate_batch(file4train)

        feed_dict = {train_inputs_a: batch_inputs_a, train_inputs_b: batch_inputs_b, train_labels: batch_labels,
                     train_freqs: freqs, train_weight: o2weight}
        run_metadata = tf.RunMetadata()

        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)

        writer.add_summary(summary, step)
        average_loss += loss_val

        if step % 10000 == 0:
            if step > 0:
                average_loss /= 10000
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
        # if epoch_index == print_index:
        #     np.savetxt('node_embeddings_temp.txt', normalized_embeddings.eval(), '%.8f')
        #     with open('node_embeddings_temp.txt', 'r') as f_r:
        #         with open('./embeddings/' + dataset_name + '/mg2vec_node_embeddings_' + str(alpha) + '_' + str(
        #                 embedding_size) + '_' + str(epoch_index - 2) + '.emb', 'w') as f_w:
        #             lines = f_r.readlines()
        #             for i in range(node_num):
        #                 temp = index2node[i] + ' ' + lines[i]
        #                 f_w.write(temp)
        #     os.remove('node_embeddings_temp.txt')
        #     print_index += 1
    final_embeddings = normalized_embeddings.eval()
    final_m_embeddings = normalized_m_embeddings.eval()
    # Save the model for checkpoints.
    saver.save(session, os.path.join('log', 'model.ckpt'))
    np.savetxt('node_embeddings_temp.txt', final_embeddings, '%.8f')
    with open('node_embeddings_temp.txt', 'r') as f_r:
        with open('./embeddings/' + dataset_name + '/mg2vec_node_embeddings_mg_3.emb', 'w') as f_w:
            lines = f_r.readlines()
            for i in range(node_num):
                temp = index2node[i] + ' ' + lines[i]
                f_w.write(temp)
    os.remove('node_embeddings_temp.txt')

    # np.savetxt('mg_embeddings_temp.txt', final_m_embeddings, '%.8f')
    # with open('mg_embeddings_temp.txt', 'r') as f_r:
    #     with open('./embeddings/' + dataset_name + '/mg2vec_mg_embeddings'+str(alpha)+'.emb', 'w') as f_w:
    #         lines = f_r.readlines()
    #         for i in range(mg_num):
    #             temp = index2mg[i] + ' ' + lines[i]
    #             f_w.write(temp)
    # os.remove('mg_embeddings_temp.txt')
writer.close()
