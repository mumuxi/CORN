import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
import utils as utils
from time import time
import json
import os
import random

#dataset
USER_GROUP = 'user_group_seq.dat'
U_I_INTERACTION = 'u_i_interaction.dat'
DATESET_PROFILE = 'date_factor.rec'
USER_INFO = 'user_info.dat'
ITEM_INFO = 'item_info.dat'


FM_FLAG = True
WINDOW_SIZE = 20
COMPONENT_SIZE = 4
COMPONENT_RANGE = [1,2,3,4]
EMBEDDING_SIZE = 64
CONVOLUTION_SIZE = 16
ATTENTION_SIZE = 32
ACTIVATION_TYPE = 'tanh'
LOSS_TYPE = 'log_loss'
LEARNING_RATE = 0.001
LAMDA = 1e-6
OPTIMIZER_TYPE = 'adam'
EPOCH = 30
BATCH_SIZE = 64
BN_FLAG = True
VERBOSE = 1
EPSILON = 1e-7
KEEP_OUT = 0.8
KS = [1,5,10]
SPLIT_N = 1
FEATURE_WIDTH = 8 # 10 for tafang  8 for ML-1M

NEGATIVE_SAMPLES = 2
START_POINT = 0

class Process():


    def __init__(self, path):
        self.path = path

        date_factor_rec = os.path.join(self.path, 'date_factor.rec')
        date_dict_rec = os.path.join(self.path, 'date_dict.rec')
        with open(date_dict_rec, 'r') as date_recorder:
            self._date_index = json.loads(date_recorder.readline())

        with open(date_factor_rec, 'r') as date_factor_recorder:
            self.factor_recorder = json.loads(date_factor_recorder.readline())

        self.user_size = self.factor_recorder['user_size']
        self.item_size = self.factor_recorder['item_size']
        # self.field_size = self.factor_recorder['field_size']
        self.feature_size = 0
        self.rated_items = [set() for _ in range(self.factor_recorder['user_size'])]
        self.unrated_items = [[]] * self.factor_recorder['user_size']
        self.user_record = dict()

    def feedCRN(self):

        self.user_info_dict = dict()
        self.item_info_dict = dict()
        test_items = set()
        user_info_file = os.path.join(self.path, USER_INFO)
        for line in open(user_info_file, encoding='utf-8'):
            term = json.loads(line)
            self.user_info_dict.update({term[0]: term})

        item_info_file = os.path.join(self.path, ITEM_INFO)
        for line in open(item_info_file, encoding='utf-8'):
            term = json.loads(line)
            self.item_info_dict.update({term[0] - self.user_size: term})

        training_seq = []
        training_Y = []
        training_user = []
        training_ui = []

        testing_seq = []
        testing_Y = []
        testing_user = []
        testing_ui = []

        user_seq_file = os.path.join(self.path, USER_GROUP)
        base = max(self._date_index)
        max_incre = 1
        for line in open(user_seq_file):
            uid, seq = json.loads(line)
            self.user_record[uid] = len(seq)
            if len(seq) < 1:
                continue
            ins_seq = []
            cur_t = seq[0][0]

            self.rated_items[uid].add(seq[0][1] - self.user_size)
            for index in range(len(seq) - 1):
                if len(ins_seq) == WINDOW_SIZE:
                    ins_seq.pop(0)
                interV = int(np.log(seq[index][0] - cur_t + 1)) + 1
                if max_incre < interV:
                    max_incre = interV
                ins_seq.append([seq[index][1], self._date_index[utils.timestampToWallClock_2(seq[index][0])], base + interV])#
                cur_t = seq[index][0]
                if index < START_POINT:
                    self.rated_items[uid].add(seq[index][1] - self.user_size)
                    continue
                cur_y = seq[index + 1][1]
                cur_date = self._date_index[utils.timestampToWallClock_2(seq[index + 1][0])]
                cur_interV = int(np.log(seq[index + 1][0] - cur_t + 1)) + 1
                if max_incre < cur_interV:
                    max_incre = cur_interV
                user = [uid, cur_date, cur_interV + base]
                ui = np.concatenate([self.user_info_dict[uid], self.item_info_dict[cur_y - self.user_size], [cur_date, base + cur_interV]])#, base + cur_interV

                if (index + 1 < len(seq) - SPLIT_N):
                    self.rated_items[uid].add(cur_y - self.user_size)
                    training_seq.append(ins_seq.copy())
                    training_Y.append([cur_y - self.user_size])
                    training_user.append(user)
                    training_ui.append(ui)
                else:
                    testing_seq.append(ins_seq.copy())
                    testing_Y.append([cur_y - self.user_size])
                    testing_user.append(user)
                    testing_ui.append(ui)
                    test_items.add(cur_y - self.user_size)
        #training['seq'], training['ui'], training['Y'], training['rate']
        self.feature_size = np.max(self._date_index) + 1 + max_incre
        self.test_items = np.array(list(test_items), dtype=np.int64)
        for i in range(len(self.rated_items)):
            self.unrated_items[i] = list(set(range(self.item_size)) - self.rated_items[i])
        return Datablock(training_seq, training_Y, training_user, training_ui), \
               Datablock(testing_seq, testing_Y, testing_user, testing_ui)

class Datablock():

    def __init__(self, seq, targets, user, interaction):
        self.seq = seq
        self.target = np.array(targets, np.int64)
        self.user = np.array(user, np.int64)
        self.interaction = np.array(interaction, np.int64)


class CORN():

    def __init__(self, process, component_size, embedding_size, convolution_size,
                 attention_size, activation_type, loss_type, learning_rate, lamda, optimizer_type, epoch, batch_size):
        self.feature_size = process.feature_size
        self.item_size = process.item_size
        self.component_size = component_size
        self.embedding_size = embedding_size
        self.convolution_size = convolution_size
        self.attention_size = attention_size
        if activation_type == 'tanh':
            self.activation = tf.nn.tanh
        elif activation_type == 'relu':
            self.activation = tf.nn.relu
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.optimizer_type = optimizer_type
        self.epoch = epoch
        self.batch_size = batch_size
        self.concat_lay_size = int (FEATURE_WIDTH * (FEATURE_WIDTH - 1) / 2 + FEATURE_WIDTH + self.embedding_size)#

        print("CORN: feature_size=%d, item_size=%d, component_size=%d, embedding_size=%d, convolution_size=%d,#epoch=%d, "
              "batch=%d, loss_type=%s, lr=%.4f, lambda=%.1e, optimizer=%s, activation=%s"
            % (self.feature_size, self.item_size, component_size, embedding_size, convolution_size, epoch, batch_size, loss_type,
               learning_rate, lamda, optimizer_type, activation_type))

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            tf.set_random_seed(2018)

            self.train_phrase = tf.placeholder(tf.bool, name='training_phrase')
            self.sequence = tf.placeholder(tf.int32, shape=[None, None, None], name='sequence_ph')
            self.mask = tf.placeholder(tf.float32, shape=[None, None], name='mask')
            self.target = tf.placeholder(tf.int32, shape=[None, None], name='labels')
            self.rate = tf.placeholder(tf.float32, shape=[None, None], name='rate')
            self.input_ui = tf.placeholder(tf.int32, shape=[None, None, None], name='ui_info_matrix')
            self.user = tf.placeholder(tf.int32, shape=[None, None], name='user')
            self.keep_out = tf.placeholder(tf.float32, name="dropout_keep")

            self.weights = self._init_weights()

            with tf.name_scope('embedding_layer'):
                sequence_emb = tf.nn.embedding_lookup(self.weights['embedding'], self.sequence)
                ui_interaction = tf.nn.embedding_lookup(self.weights['embedding'], self.input_ui)
                target_emb = tf.nn.embedding_lookup(self.weights['label_embedding'], self.target)
                # user_emb = tf.nn.embedding_lookup(self.weights['embedding'], self.user[:,0])
                # cur_context_emb = tf.nn.embedding_lookup(self.weights['embedding'], self.user[:,1:])
                # cur_context_emb = tf.reduce_sum(cur_context_emb, axis=1)

            with tf.name_scope('ui_interaction'):
                interactions = [tf.reduce_sum(ui_interaction, axis=-1)]# add first order feature information#
                for i in range(FEATURE_WIDTH):
                    for j in range(i + 1, FEATURE_WIDTH):
                        interact = tf.reduce_sum(tf.multiply(ui_interaction[:,:,i,:], ui_interaction[:,:,j,:]), axis=-1, keep_dims=True)
                        interactions.append(interact)
                interactions = tf.concat(interactions, axis=-1)
                interactions = tf.nn.dropout(interactions, self.keep_out)

            tempor_emb = tf.reduce_sum(sequence_emb[:, :, 1:, :], axis=-2)
            # tempor_emb = tf.add(tempor_emb, tf.ones_like(tempor_emb, dtype=tf.float32))
            item_seq = tf.add(tempor_emb, sequence_emb[:, :, 0, :])

            components = []
            for width in COMPONENT_RANGE:
                filter = tf.Variable(
                    tf.truncated_normal([width, self.embedding_size, self.convolution_size], stddev=0.1))
                component = tf.nn.tanh(
                    tf.add(tf.nn.conv1d(item_seq, filter, stride=1, padding='SAME'), self.weights['convolutional_b']))
                attention = self.self_attention(component, self.mask)
                conv = tf.multiply(component, attention)
                components.append(tf.reduce_sum(conv, axis=-2))  # - width + 1
                # conv = tf.reduce_max(component, axis=-2)
                # components.append(conv)

            convolution_output = tf.concat(components, axis=-1)
            convolution_output = tf.nn.dropout(convolution_output, self.keep_out)
            convolution_output = tf.add(tf.matmul(convolution_output,self.weights['conv_out_layer']), self.weights['conv_out_bias'])
            # convolution_output = self.batch_norm(convolution_output, self.train_phrase, 'convolutional_output')
            convolution_output = self.activation(convolution_output)
            convolution_output = tf.multiply(target_emb, tf.expand_dims(convolution_output, axis=-2))

            concat = tf.concat([tf.nn.l2_normalize(interactions, dim=-1), tf.nn.l2_normalize(convolution_output, dim=-1)], axis=-1)

            output_w = tf.Variable(np.random.normal(loc=0.0, scale=np.sqrt(2.0 / (self.concat_lay_size + 1)),
                                                               size=(self.concat_lay_size, 1)), dtype=tf.float32)
            output_b = tf.Variable(tf.constant(0.0), dtype=tf.float32)
            output_w = tf.add(output_w, tf.ones_like(output_w))
            self.visual = output_w

            self.out = tf.add(tf.matmul(tf.reshape(concat, [-1, self.concat_lay_size]), output_w), output_b)

            if self.loss_type == 'log_loss':
                # self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(tf.reshape(self.rate, [-1, 1]), tf.reshape(tf.nn.sigmoid(self.out), [-1,1]))
                for _, variables in self.weights.items():
                    self.loss += tf.contrib.layers.l2_regularizer(self.lamda)(variables)
                self.loss += tf.contrib.layers.l2_regularizer(self.lamda)(tf.subtract(output_w, tf.ones_like(output_w)))
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.95).minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.sess = self._init_sess()
            self.sess.run(tf.global_variables_initializer())
            tf.summary.FileWriter('CORN/log', self.graph)

    def self_attention(self, input, mask):
        with tf.name_scope('self_attention'):
            new_input = tf.reshape(input, shape=[-1, self.convolution_size])
            attention = tf.matmul(tf.nn.tanh(tf.add(tf.matmul(new_input, self.weights['attention_w']), self.weights['attention_b'])),
                                  self.weights['attention_h'])
            attention = tf.multiply(tf.reshape(tf.exp(attention), shape=[-1, WINDOW_SIZE]), mask)
            attention = tf.div(attention, tf.reduce_sum(attention, axis=-1, keep_dims=True))
            return tf.expand_dims(attention, -1)

    def _init_sess(self):
        config = tf.ConfigProto()  # device_count={"gpu": 0}
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def batch_norm(self, x, training_phrase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(training_phrase , lambda: bn_train, lambda: bn_inference)
        return z

    def _init_weights(self):
        weights = dict()
        weights['embedding'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), dtype=tf.float32, name='embedding')
        weights['label_embedding'] = tf.Variable(tf.random_normal([self.item_size, self.embedding_size], 0.0, 0.01), dtype=tf.float32, name='label_embedding')
        weights['feature_bias'] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 0.1), dtype=tf.float32, name='feature_bias')
        weights['bias'] = tf.Variable(tf.constant(0.0))
        weights['convolutional_b'] = tf.Variable(tf.random_uniform([1, self.convolution_size], 0.0, 0.1), dtype=tf.float32)
        weights['conv_out_layer'] = tf.Variable(tf.random_normal([self.component_size * self.convolution_size, self.embedding_size], 0.0, 1.0 / self.embedding_size), dtype=tf.float32, name='conv_out_layer')
        weights['conv_out_bias'] = tf.Variable(tf.random_uniform([1, self.embedding_size], 0.0, 1.0 / self.embedding_size), dtype=tf.float32, name='conv_out_bias')
        #np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor[1], self.hidden_factor[0])
        weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=np.sqrt(2.0 / (self.attention_size + 1)),
                                                              size=(self.attention_size, 1)),dtype=tf.float32)
        weights['attention_w'] = tf.Variable(np.random.normal(loc=0, scale=np.sqrt(2.0 / (self.convolution_size + self.attention_size)),
                                                              size=(self.convolution_size, self.attention_size)), dtype=tf.float32, name='attention_weight')
        weights['attention_b'] = tf.Variable(tf.random_uniform([1, self.attention_size], 0.0, 0.1), dtype=tf.float32, name='attention_out')
       #concat_weights:
        return weights

    def train(self, training, testing):
        # if VERBOSE:
        #     test_p, test_r, test_m = self.evaluate(testing)
        #     print("Initial: prec=[%.4f,%.4f,%.4f], "
        #           "recall=[%.4f,%.4f,%.4f], "
        #           "map=[%.4f]"
        #           % (test_p[0], test_p[1], test_p[2],
        #              test_r[0], test_r[1], test_r[2],
        #              test_m))

        for epoch in range(self.epoch):
            t1 = time()

            user = training.user.copy()
            seq = training.seq.copy()
            interaction = training.interaction.copy()
            target = training.target.copy()

            neg_samples = self.gene_negative_samples(user[:,0])
            labels = np.concatenate([target, neg_samples], axis=-1)

            neg_interaction = self.gene_negative_interaction(neg_samples, user[:,0], interaction)
            interaction = np.concatenate([np.expand_dims(interaction, axis=1), neg_interaction], axis=1)

            rates = np.concatenate([np.ones_like(target, np.float32), np.zeros_like(neg_samples, np.float32)], axis=-1)

            self.shuffle_in_unison_scary(user, seq, interaction, labels)
            batch_count = int ((len(labels) + self.batch_size) / self.batch_size)
            total_loss = 0
            for batch in range(batch_count):
                _seq, _mask, _Y, _rate, _user, _interact = self.get_batch(seq, labels, user, rates, interaction, batch)
                feed_dict = {
                    self.train_phrase: True,
                    self.sequence: _seq,
                    self.input_ui: _interact,
                    self.mask: _mask,
                    self.target: _Y,
                    self.user: _user,
                    self.rate: _rate,
                    self.keep_out: KEEP_OUT
                }
                loss, visual, _ = self.sess.run([self.loss, self.visual, self.optimizer], feed_dict=feed_dict)
                # print(batch)
                total_loss += loss
            total_loss /= batch_count
            t2 = time()
            # evaluate training and validation datasets
            # print(visual)
            test_p, test_r, test_m = self.evaluate(testing)

            if VERBOSE:
                print("Epoch (logloss,ROC,accuracy) %d [%.1f s]\tprec=[%.4f,%.4f,%.4f], "
                      "recall=[%.4f,%.4f,%.4f], "
                      "map=[%.4f] [%.1f s]"
                      "loss=%.4f"
                      % (epoch + 1, t2 - t1, test_p[0], test_p[1], test_p[2],
                         test_r[0], test_r[1], test_r[2],
                         test_m, time() - t2, total_loss))

    def evaluate(self, data):
        batch_count = int ((len(data.target) + self.batch_size - 1) / self.batch_size)
        seq = data.seq
        user = data.user
        interaction = data.interaction

        # test_item = process.test_items.copy().tolist()
        # np.random.shuffle(test_item)
        #
        # labels = [test_item[:1000]] * len(data.target)  # for test items
        # # labels = [list(range(process.item_size))] * len(data.target)  # #for all items
        # for i in range(len(data.target)):
        #     if data.target[i] not in labels[i]:
        #         labels[i].pop(0)
        #         labels[i].append(data.target[i])
        # labels = np.array(labels, np.int64)
        # rates = np.ones_like(labels, np.float32)
        labels = [process.test_items] * len(data.target)
        labels = np.array(labels)
        split_size = 1000
        split_count = int((len(process.test_items) + split_size - 1) / split_size)
        start = 0
        end = split_size
        y_pred = None
        for i in range(split_count):
            split_labels = labels[:,start:end]
            split_interaction = self.gene_negative_interaction(split_labels, user[:,0], interaction)
            rates = np.ones_like(split_labels, np.float32)

            t_pred = None

            for batch in range(batch_count):
                _seq, _mask, _Y, _rate, _user, _interact = self.get_batch(seq, split_labels, user, rates, split_interaction, batch)
                feed_dict = {
                    self.train_phrase: False,
                    self.sequence: _seq,
                    self.mask: _mask,
                    self.input_ui: _interact,
                    self.target: _Y,
                    self.rate: _rate,
                    self.user: _user,
                    self.keep_out: 1.0
                }
                temp_pred = self.sess.run(self.out, feed_dict=feed_dict)
                if batch == 0:
                    t_pred = temp_pred
                else:
                    t_pred = np.concatenate([t_pred, temp_pred], axis=0)
            if i == 0:
                y_pred = np.reshape(t_pred, [-1,end - start])
            else:
                y_pred = np.concatenate([y_pred, np.reshape(t_pred, [-1,end - start])], axis=-1)
            start = end
            end += split_size
            end = len(process.test_items) if end > len(process.test_items) else end
        y_pred = -1 * np.reshape(y_pred, newshape=[-1,len(labels[0])])
        prec, rec, maps = self.rank_based_evaluate(data.target, y_pred, KS, user[:,0], process.rated_items, labels)
        # prec, rec, maps = self.rank_based_evaluate(data.target, -y_pred, KS, user[:, 0], process.rated_items,
        #                                            np.array(range(process.item_size)))#for all items
        return prec, rec, maps

    def rank_based_evaluate(self, y_true, y_pred, k, users, rated, itemsets):
        if not isinstance(k, list):
            ks = [k]
        else:
            ks = k

        precisions = [list() for _ in range(len(ks))]
        recalls = [list() for _ in range(len(ks))]
        apks = list()

        for i in range(len(y_true)):
            pred = np.array(y_pred[i]).argsort()
            pred = itemsets[i][np.array(pred, np.int64)]
            pred = [p for p in pred if p not in rated[users[i]]]
            for j, _k in enumerate(ks):
                prec, rec = utils._compute_precision_recall(y_true[i], pred, _k)
                precisions[j].append(prec)
                recalls[j].append(rec)
            apks.append(utils._compute_apk(y_true[i], pred, np.inf))

        #record hit_rate
        user_hit_5, user_hit_10, user_hit_20 = [], [], []
        def rescale(user):
            record = process.user_record[user] - 20
            if record <= 5:
                return 0
            elif record <= 10:
                return 1
            elif record <= 20:
                return 2
            else:
                return 3
        hit_rate = np.array(recalls)
        for i in range(len(users)):
            scale = rescale(users[i])
            if scale == 0:
                user_hit_5.append(hit_rate[:,i])
            elif scale == 1:
                user_hit_10.append(hit_rate[:,i])
            elif scale == 2:
                user_hit_20.append(hit_rate[:,i])
        print("hit_rate for [5,10,20]:", np.mean(user_hit_5, axis=0), np.mean(user_hit_10, axis=0), np.mean(user_hit_20, axis=0))
        return np.mean(precisions, axis=-1), np.mean(recalls, axis=-1), np.mean(apks)

    def early_stop(self, valid_result):
        if len(valid_result) > 5:
            if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4] > valid_result[-5]:
                return True
        return False

    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def get_batch(self, seq, Y, user, rate, interaction, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        end = end if end < len(Y) else len(Y)
        x_batch = seq[start:end]
        seq_len = [len(term) for term in x_batch]
        max_len = np.max(seq_len)
        for index in range(len(x_batch)):
            x_batch[index] = x_batch[index] + [[0] * len(x_batch[index][0])] * (max_len - len(x_batch[index]))
        weight_mask = sequence_mask(seq_len, max_len)
        return x_batch, weight_mask, Y[start:end], rate[start:end], user[start:end], interaction[start:end]

    def gene_negative_samples(self, users):
        neg_samples = np.zeros([len(users), NEGATIVE_SAMPLES], dtype=np.int64)
        for i in range(len(users)):
            uid = users[i]
            samples = random.sample(set(range(process.item_size)) - process.rated_items[uid], NEGATIVE_SAMPLES)
            for j in range(len(samples)):
                neg_samples[i, j] = samples[j]
        return neg_samples

    def gene_negative_interaction(self, neg_samples, users, interaction):
        neg_interactions = [list() for _ in range(len(neg_samples))]

        user_map = process.user_info_dict
        item_map = process.item_info_dict

        for i in range(len(neg_samples)):
            for j in range(len(neg_samples[i])):
                interact = np.concatenate([user_map[users[i]], item_map[neg_samples[i][j]], interaction[i][-2:]])
                neg_interactions[i].append(interact)

        return neg_interactions

def sequence_mask(seq_len, max_seq_len):
    mask = np.zeros([len(seq_len), max_seq_len], dtype=float)
    for (index, value) in enumerate(seq_len):
        for i in range(value):
            mask[index, i] = 1.0
    return mask

def constructor(process):
    model = CORN(process, COMPONENT_SIZE, EMBEDDING_SIZE,  CONVOLUTION_SIZE, ATTENTION_SIZE,
                    ACTIVATION_TYPE, LOSS_TYPE, LEARNING_RATE, LAMDA, OPTIMIZER_TYPE, EPOCH, BATCH_SIZE)
    return model

if __name__ == '__main__':
    path = '/data/qzhang6/data/MovieLens-1M/'
    process = Process(os.path.join(path, '20_20_1'))#_Number
    train, test = process.feedCRN()
    CRNN = constructor(process)
    CRNN.train(train, test)