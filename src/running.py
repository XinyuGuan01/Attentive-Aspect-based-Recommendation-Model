# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from time import time
import numpy as np
import pandas as pd
import math
import random
import heapq
import argparse
from gensim.models import KeyedVectors
from collections import defaultdict
from aarm import AARM

def parse_args():
    parser = argparse.ArgumentParser(description="Run AARM model")
    parser.add_argument('--process_name', nargs='?', default="aarm", help='the name of process')
    parser.add_argument('--gpu', type=int, default=0, help="Specify which GPU to use (default=0)")

    # dataset
    parser.add_argument('--productName', nargs='?', default='Beauty', help='specify the dataset used in experiment')
    parser.add_argument('--MaxPerUser', type=int, default=12, help='Maximum number of aspects per user.')
    parser.add_argument('--MaxPerItem', type=int, default=15, help='Maximum number of aspects per item.')
    parser.add_argument('--auto_max', type=int, default=1, help='1: set MaxPerUser (MaxPerItem) as the 75% quantile of the sizes of all user (item) aspect set; 0: using the args of MaxPerUser and MaxPerItem')

    # regularization
    parser.add_argument('--is_l2_regular', type=int, default=0, help='1: use l2_regularization for embedding matrix')
    parser.add_argument('--is_out_l2', type=int, default=0, help='1: to use l2 regularization in output layer')
    parser.add_argument('--lamda_l2', type=float, default=0.1, help='parameter of the l2_regularization for embedding matrix')
    parser.add_argument('--lamda_out_l2', type=float, default=0.1,help='parameter of the l2_regularization for output layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='the keep probability of dropout')

    # training
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--optimizer_type', type=int, default=1, help="1: AdamOptimizer, 2: AdagradOptimizer, 3: GradientDescentOptimizer, 4:MomentumOptimizer")
    parser.add_argument('--num_aspect_factor', type=int, default=128, help='dimension of aspect embedding')
    parser.add_argument('--num_mf_factor', type=int, default=128, help='dimension of global user/item embedding')
    parser.add_argument('--num_attention', type=int, default=64, help='dimension of attention module')
    parser.add_argument('--num_batch', type=int, default=1024,help='batch size of training')
    parser.add_argument('--is_save_params', type=int, default=1, help='1: save the parameters which achieved the best validation performance.')
    parser.add_argument('--patience_no_improve', type=int, default=300, help="The number of patience epochs. The training would be stopped if half of the four measures decreased for 'patience_no_improve' success epochs")
    parser.add_argument('--seed', type=int, default=1000, help='random seed')

    # evaluate
    parser.add_argument('--K', type=int, default=10, help='top-K recommendation.')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='batch size of test evaluation')

    return parser.parse_args()

class Running():
    def __init__(self, args):
        self.args = args
        self.productName = self.args.productName

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True      
        self.sess = tf.Session(config=config)

        self.load_dataset()
        self.load_emb()

        # initialize AARM models
        self.model = AARM(args, self.num_user, self.num_item, self.num_aspect, self.MaxPerUser, self.MaxPerItem, self.user_aspect_padded, 
                    self.item_aspect_padded, self.aspect_vectors)
        self.model.build_graph()
        self.save_path_name = "../temp/%s/" % (args.process_name)
        self.restore_path_name = "../temp/%s/" % (args.process_name)
        if not os.path.exists(self.save_path_name):
            os.makedirs(self.save_path_name)
        if not os.path.exists(self.restore_path_name):
            os.makedirs(self.restore_path_name)
        self.saver = tf.train.Saver()

        # initialize graph
        self.sess.run(tf.global_variables_initializer())

        # number of params
        total_parameters = 0
        for variable in self.model.all_weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

    #### training
    def _get_train_batch(self, iter):
        user_input = []
        item_p_input = []
        item_n_input = []
        for index in range(iter * self.args.num_batch, min(len(self.train_pairs), (iter + 1) * self.args.num_batch)):
            u, i = self.train_pairs[index]
            user_input.append(u)
            item_p_input.append(i)
            j = np.random.randint(self.num_item)
            while j in self.user2items_train[u]:
                j = np.random.randint(self.num_item)
            item_n_input.append(j)
        return (user_input, item_p_input, item_n_input)

    def generate_train_batch(self):
        num_iters = len(self.train_pairs) // self.args.num_batch + 1
        user_list = []
        item_pos_list = []
        item_dns_list = []
        for iter in range(num_iters):
            u_l, v_pos_l, v_neg_l = self._get_train_batch(iter)
            user_list.append(u_l)
            item_pos_list.append(v_pos_l)
            item_dns_list.append(v_neg_l)

        return user_list, item_pos_list, item_dns_list, num_iters

    def train(self):
        self.best_result = [(0, 0, 0, 0)]  # [(precision, recall, ndcg, hr)]
        self.counter_no_improve = 0 # how many epochs that no improvements on ranking measures
        for epoch in range(self.args.num_epoch):
            self.counter_no_improve += 1
            if self.counter_no_improve > self.args.patience_no_improve:
                break
            t1 = time()
            loss_list = []
            trainBpr_batch = []
            user_list, item_pos_list, item_dns_list, num_iters = self.generate_train_batch()
            for i in range(num_iters):
                user_batch, item_p_batch, item_n_batch = user_list[i], item_pos_list[i], item_dns_list[i]
                feed_dict = {self.model.user_input:user_batch, self.model.item_p_input:item_p_batch, self.model.item_n_input:item_n_batch,
                self.model.dropout_keep:args.dropout}
                loss_, bprloss_, opt = self.sess.run((self.model.loss, self.model.bprloss, self.model.train_op), feed_dict=feed_dict)
                loss_list.append(loss_)
                trainBpr_batch.append(bprloss_)

            train_loss = np.mean(loss_list)
            train_bprloss = np.mean(trainBpr_batch) 
            if (epoch+1) % 10 == 0:
                t2 = time()
                precision, recall, ndcg, hr = self.evaluate_model(self.users4valid, mode='valid')
                print("Epoch %d [%.1f s]\tloss=%.6f\tbprloss=%.6f" % (epoch + 1, t2 - t1, train_loss, train_bprloss))
                print("validation: precision=%.6f\trecall=%.6f\tNDCG=%.6f\tHT=%.6f [%.1f s]"
                    % (precision, recall, ndcg, hr, time() - t2))
                if self.args.is_save_params:
                    for p,r,n,h in self.best_result:
                        if np.sign(precision-p) + np.sign(recall-r) + np.sign(ndcg-n) + np.sign(hr-h) >= 0:
                            self.counter_no_improve = 0
                            self.best_result = [(precision, recall, ndcg, hr)]
                            save_path = self.saver.save(self.sess, self.save_path_name+'save_net.ckpt')
                            print("epoch %d save to %s" % (epoch+1, save_path))
    
    def test(self):
        if self.args.is_save_params:
            self.saver.restore(self.sess, self.restore_path_name+'save_net.ckpt')

        t3 = time()
        precision, recall, ndcg, hr = self.evaluate_model(range(self.num_user), mode='test')
        print("test_precision=%.6f\ttest_recall=%.6f\ttest_NDCG=%.6f\ttest_HT=%.6f [%.1f s]"
            % (precision, recall, ndcg, hr, time() - t3))
        print("MaxPerUser is %d, MaxPerItem is %d" %(self.MaxPerUser, self.MaxPerItem))

    #### load datasets
    def load_dataset(self):
        data_path = '../data/'+self.productName
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        tf.set_random_seed(self.args.seed)

        # load training and test pairs
        self.train_pairs = []
        with open(data_path+'/train_pairs.txt') as f:
            for line in f:
                self.train_pairs.append(list(map(int, line.split(','))))
        perm = np.random.permutation(range(len(self.train_pairs)))
        self.train_pairs = np.array(self.train_pairs)[perm]

        self.valid_pairs = []
        with open(data_path+'/valid_pairs.txt') as f:
            for line in f:
                self.valid_pairs.append(list(map(int, line.split(','))))
        perm = np.random.permutation(range(len(self.valid_pairs)))
        self.valid_pairs = np.array(self.valid_pairs)[perm]

        self.test_pairs = []
        with open(data_path+'/test_pairs.txt') as f:
            for line in f:
                self.test_pairs.append(list(map(int, line.split(','))))
        perm = np.random.permutation(range(len(self.test_pairs)))
        self.test_pairs = np.array(self.test_pairs)[perm]

        # load id2user, id2item, id2aspect
        with open(data_path+'/users.txt') as users_f:
            self.id2user = {}
            self.user2id = {}
            index = 0
            for line in users_f:
                name = line.strip()
                self.id2user[index] = name
                self.user2id[name] = index
                index += 1
        with open(data_path+'/product.txt') as products_f:
            self.id2item = {}
            self.item2id = {}
            index = 0
            for line in products_f:
                name = line.strip()
                self.id2item[index] = name
                self.item2id[name] = index
                index += 1
        with open(data_path+'/aspect.txt') as f:
            self.id2aspect = {}
            self.aspect2id = {}
            index = 0
            for line in f:
                name = line.strip()
                self.id2aspect[index] = name
                self.aspect2id[name] = index
                index += 1
        self.num_user = len(self.id2user)
        self.num_item = len(self.id2item)
        self.num_aspect = len(self.id2aspect)
        self.item_set = set(self.id2item.keys())
        
        # generate user2items, dict: {u:[v1,v2,...], ...}
        self.user2items_train = defaultdict(list)
        self.user2items_valid = defaultdict(list)
        self.user2items_test = defaultdict(list)

        for u,v in self.train_pairs:
            self.user2items_train[u].append(v)
        
        self.users4valid = []
        for u,v in self.valid_pairs:
            self.user2items_valid[u].append(v)
            self.users4valid.append(u)

        for u,v in self.test_pairs:
            self.user2items_test[u].append(v)

        # load ranked user/item aspect sets
        self.user_aspect_rank_dict = {}
        with open(data_path+'/user_aspect_rank.txt') as f:
            index = 0
            for line in f:
                if line.strip():
                    tokens = [int(t) for t in line.strip().split(',')]
                    self.user_aspect_rank_dict[index] = tokens
                else:
                    self.user_aspect_rank_dict[index] = []
                index += 1

        self.item_aspect_rank_dict = {}
        with open(data_path+'/item_aspect_rank.txt') as f:
            index = 0
            for line in f:
                if line.strip():
                    tokens = [int(t) for t in line.strip().split(',')]
                    self.item_aspect_rank_dict[index] = tokens
                else:
                    self.item_aspect_rank_dict[index] = []
                index += 1
        self.get_history_aspect()

    def get_history_aspect(self):
        user_aspect_list = [[]] * self.num_user
        item_aspect_list = [[]] * self.num_item

        len_users = [len(self.user_aspect_rank_dict[u]) for u in self.user_aspect_rank_dict]
        len_items = [len(self.item_aspect_rank_dict[v]) for v in self.item_aspect_rank_dict]

        lens_u_series = pd.Series(len_users)
        lens_v_series = pd.Series(len_items)

        if self.args.auto_max:
            self.MaxPerUser = int(lens_u_series.quantile(0.75))
            self.MaxPerItem = int(lens_v_series.quantile(0.75))
        else:
            self.MaxPerUser = self.args.MaxPerUser
            self.MaxPerItem = self.args.MaxPerItem
        print("MaxPerUser is %d, MaxPerItem is %d" %(self.MaxPerUser, self.MaxPerItem))

        for u in self.user_aspect_rank_dict.keys():
            user_aspect_list[u] = self.user_aspect_rank_dict[u]
        for v in self.item_aspect_rank_dict.keys():
            item_aspect_list[v] = self.item_aspect_rank_dict[v]

        self.user_aspect_padded = np.zeros((self.num_user, self.MaxPerUser), dtype=np.int32)
        self.item_aspect_padded = np.zeros((self.num_item, self.MaxPerItem), dtype=np.int32)

        for idx, s in enumerate(user_aspect_list):
            trunc = np.asarray(s[:self.MaxPerUser])
            self.user_aspect_padded[idx, :len(trunc)] = trunc
        for idx, s in enumerate(item_aspect_list):
            trunc = np.asarray(s[:self.MaxPerItem])
            self.item_aspect_padded[idx, :len(trunc)] = trunc

    def load_emb(self):
        embedding = KeyedVectors.load_word2vec_format('../data/' + self.productName + '/emb' + str(self.args.num_aspect_factor) + '.vector')

        # construct a dict from vocab's index to embedding's index
        embed_dict = {}  # {word:index,...}
        for index, word in enumerate(embedding.index2word):
            embed_dict[word] = index

        self.aspect_vectors = np.zeros((self.num_aspect, self.args.num_aspect_factor),dtype=np.float32)

        trained_aspects = []
        untrained_aspects = []
        aspect_set = self.aspect2id.keys()
        for a in aspect_set:
            a_ = '_'.join(a.split())
            if a_ in embed_dict:
                trained_aspects.append(a)
                self.aspect_vectors[self.aspect2id[a]] = embedding.syn0[embed_dict[a_]]
            else:
                untrained_aspects.append(a)
        print('trained aspects: %d, untrained aspects: %d' % (len(trained_aspects), len(untrained_aspects)))

    #### evaluate
    def eval_one_user(self, u, mode):
        if mode == 'valid':
            gtItems = self.user2items_valid[u]
        elif mode == 'test':
            gtItems = self.user2items_test[u]
        test_neg = list(self.item_set - set(self.user2items_train[u]) - set(gtItems)) 

        item_list = gtItems + test_neg
        permut = np.random.permutation(len(item_list))
        item_list = np.array(item_list, dtype=np.int32)[permut]
        user_list = np.full(len(item_list),u,dtype='int32')

        num_batch = len(item_list) // self.args.test_batch_size
        predictions = []
        for i in range(num_batch+1):
            feed_dict = {self.model.user_input: user_list[(i)*self.args.test_batch_size:(i+1)*self.args.test_batch_size],
                        self.model.item_p_input: item_list[(i)*self.args.test_batch_size:(i+1)*self.args.test_batch_size],
                        self.model.dropout_keep: 1}
            predictions.extend( list(self.sess.run((self.model.rating_preds_pos), feed_dict=feed_dict).squeeze()) )

        map_item_score = {}
        for i in range(len(item_list)):
            v = item_list[i]
            map_item_score[v] = predictions[i]

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.args.K, map_item_score, key=map_item_score.get) # top K
        recall, ndcg, hr, precision = self.metrics(ranklist, gtItems)
        return precision,recall,ndcg,hr

    def evaluate_model(self, users, mode):
        precision_list = []
        recall_list = []
        ndcg_list = []
        hr_list = []
        for u in users:
            (p,r,ndcg,hr) = self.eval_one_user(u, mode)
            precision_list.append(p)
            recall_list.append(r)
            ndcg_list.append(ndcg)
            hr_list.append(hr)
        return (np.mean(precision_list), np.mean(recall_list), np.mean(ndcg_list), np.mean(hr_list))

    def metrics(self, doc_list, rel_set):
        dcg = 0.0
        hit_num = 0.0
        for i in range(len(doc_list)):
            if doc_list[i] in rel_set:
                #dcg
                dcg += 1/(math.log(i+2)/math.log(2))
                hit_num += 1
        #idcg
        idcg = 0.0
        for i in range(min(len(rel_set),len(doc_list))):
            idcg += 1/(math.log(i+2)/math.log(2))
        ndcg = dcg/idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(doc_list)
        #compute hit_ratio
        hit = 1.0 if hit_num > 0 else 0.0
        return recall, ndcg, hit, precision

if __name__ == '__main__':
    args = parse_args()

    print("Arguments: %s" % (args))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    runner = Running(args)
    runner.train()
    runner.test()