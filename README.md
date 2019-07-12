#CORN

from utils import *
import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from itertools import groupby

NORMALIZATION = ['standard', 'uniform']


class DataModel():
    def __init__(self, path, key_column_name, **kwargs):
        '''
        {'user_id': xxx, 'time_stamp': xxx, 'label: xxx}
        '''
        self.path = path
        self.label_flag = True
        if 'label' in key_column_name:
            label_indicator = key_column_name['label']
            assert isinstance(label_indicator, str), print(
                "Type error, 'label' should be assigned a string to indicate the label columns")
            label_indicator = label_indicator.split(',')
            self.label_indicator = []
            for label in label_indicator:
                if label.strip() != '':
                    self.label_indicator.append(label.strip())
        else:
            self.label_flag = False

        self.date_flag = True
        if 'time_stamp' in key_column_name:
            self.date_indicator = key_column_name['time_stamp']
            assert isinstance(self.date_indicator, str), print(
                "Type error, 'time_stamp' should be assigned a string to indicate the timestamp column")
        else:
            self.date_flag = False

        self.userid_flag = True
        if 'user_id' in key_column_name:
            self.user_id_indicator = key_column_name['user_id']
            assert isinstance(self.user_id_indicator, str), print(
                "Type error, 'user_id' should be assigned a string to indicate the user id column")
        else:
            self.userid_flag = False

        self.normalization_method = 'uniform'
        self.threhold_user = None
        self.threhold_diffs = None
        self.seq_len = 6

        if 'threhold_user' in kwargs:
            assert isinstance(kwargs['threhold_user'], float), print(
                "Type error, arguments 'threhold_user' should be float within [0,1.0]")
            self.threhold_user = min(max(0, kwargs['threhold_user']), 1)
        if 'threhold_diffs' in kwargs:
            assert isinstance(kwargs['threhold_diffs'], float), print(
                "Type error, arguments 'threhold_diffs' should be float within [0,1.0]")
            self.threhold_diffs = min(max(0, kwargs['threhold_diffs']), 1)
        if 'nomalization_method' in kwargs:
            self.normalization_method = kwargs['nomalization_method']
            if self.normalization_method not in NORMALIZATION:
                self.normalization_method = 'uniform'
        if 'seq_len' in kwargs:
            assert isinstance(kwargs['seq_len'], int), print(
                "Type error, arguments 'seq_len' should be a positive integer")
            self.seq_len = kwargs['seq_len']
            if self.seq_len < 0:
                self.seq_len = 6

        self.numerical_data = None
        self.categorical_data = None
        self.labels = None
        self.name_index = None
        self.tokenizer_list = None
        self.reverse_tokenizer_dict = None

        self.cat_dyn_index = None
        self.cat_sta_index = None
        self.num_dyn_index = None
        self.num_sta_index = None

        self.latest_records = dict()

    def generate_dump_file(self):
        dump_file = ''.join(self.path.split('/')[-1].split('.')[:-1]) + '_' \
                    + str(self.label_indicator) + '_' \
                    + str(self.normalization_method) + '_' \
                    + str(self.threhold_diffs) + '_' \
                    + str(self.threhold_user) + '_' \
                    + str(self.seq_len)
        return dump_file

    def save(self, name='data_embedding'):
        config = dict()
        config['normalization_method'] = self.normalization_method
        config['threhold_diffs'] = self.threhold_diffs
        config['threshold_user'] = self.threhold_user
        config['seq_len'] = self.seq_len
        pickle.dump(config, open('{}.config'.format(name), 'wb'))

    def load_config(self, name=None):
        if not name:
            name = 'data_embedding'
        if not os.path.exists(name):
            print('File {} not exist!'.format(name))
        config = pickle.load(open('{}.config'.format(name), 'rb'))
        self.normalization_method = config['normalization_method']
        self.threhold_diffs = config['threhold_diffs']
        self.threhold_user = config['threshold_user']
        self.seq_len = config['seq_len']

    @dec_timer
    def _read_file(self):
        '''read data from the csv raw data file
        :param input_dim: the input dimension
        :param data_path: data path
        :param label_indicator: the index name of label column
        :param type_indicator: a list of pandas data type indicators
        :return: tuple of four numpy array numerical data, categorical data, data of other type and labels.
        '''
        if self.path.endswith('.csv'):
            data = pd.read_csv(self.path)
        else:
            try:
                raise Exception('File type wrong')
            except Exception:
                print('The program cannot solve the file type of {}'.format(self.path.split()[-1]))
        data = data.head(1000)
        
        columns_index = data.columns.values.tolist()

        numerical_data = list()
        categorical_data = list()

        label_value_dict = dict()
        if self.label_flag:
            for label_name in self.label_indicator:
                if label_name not in columns_index:
                    try:
                        raise Exception('Keys not found!')
                    except Exception:
                        print("label indicator '{}' is not in the data head list".format(self.label_indicator))
            labels = list()
            label_dict = dict()
            count = 0
            for label_name in self.label_indicator:
                columns = data.loc[:, label_name].get_values()
                unique_c = np.unique(columns)
                value_dict = dict(zip(unique_c, range(len(unique_c))))
                label_value_dict[label_name] = value_dict
                columns = np.asarray([value_dict[l] for l in columns], dtype=int)
                labels.append(columns)
                label_dict[label_name] = count
                count += 1
            data = data.drop(self.label_indicator, axis=1)
            labels = np.stack(labels, axis=1)
            self.labels = labels
            self.label_name_dict = label_dict
            self.label_value_dict = label_value_dict
        else:
            self.labels = []
            self.label_name_dict = []
            self.label_value_dict = []

        if self.userid_flag:
            if self.user_id_indicator not in columns_index:
                try:
                    raise Exception('Keys not found!')
                except Exception:
                    print("user ids indicator '{}' is not in the data head list".format(self.label_indicator))
            user_ids = data.loc[:, self.user_id_indicator].apply(hash).get_values()
            data = data.drop(self.user_id_indicator, axis=1)
        else:
            user_ids = np.asarray(range(len(data)))
        categorical_data.append(user_ids)
        numerical_data.append(user_ids)

        if self.date_flag:
            if self.date_indicator not in columns_index:
                try:
                    raise Exception('Keys not found!')
                except Exception:
                    print("indicator for date '{}' is not in the data head list".format(self.date_indicator))
            try:
                date_column = (pd.to_datetime(data.loc[:, self.date_indicator]).get_values() - np.datetime64(
                    '1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                data = data.drop(self.date_indicator, axis=1)
                categorical_data.append(date_column)
                numerical_data.append(date_column)
            except Exception:
                print("column named '{}' is not in date type".format(self.date_indicator))

        name_index = defaultdict(list)
        type_indicator = data.dtypes
        columns_index = data.columns.values.tolist()
        for i, dtype in enumerate(type_indicator):
            if str(dtype) == 'int64':
                if len(np.unique(data.iloc[:, i].values)) < 0.01 * len(data):
                    name_index['categorical'].append(columns_index[i])
                    categorical_data.append(data.iloc[:, i].get_values().astype('str'))
                else:
                    name_index['numerical'].append(columns_index[i])
                    numerical_data.append(np.asarray(data.iloc[:, i].get_values()))
            elif str(dtype) == 'categorical':
                name_index['categorical'].append(columns_index[i])
                categorical_data.append(data.iloc[:, i].get_values().astype('str'))
            elif str(dtype) == 'object':
                name_index['categorical'].append(columns_index[i])
                categorical_data.append(data.iloc[:, i].get_values().astype('str'))
            elif str(dtype) == 'float64' or str(dtype) == 'numerical':
                name_index['numerical'].append(columns_index[i])
                numerical_data.append(np.asarray(data.iloc[:, i].get_values()))
            else:
                name_index['categorical'].append(columns_index[i])
                categorical_data.append(data.iloc[:, i].get_values().astype('str'))
        numerical_data = np.stack(numerical_data, axis=1)
        self.numerical_data = numerical_data
        self.categorical_data = categorical_data
        self.name_index = name_index

    @dec_timer
    def _data_clean(self):
        generated_data = list()
        tokenizer_reverses = list()
        tokenizers = list()
        categorical_name_list = list()
        if self.date_flag:
            start_index = 2
        else:
            start_index = 1
        i = -1
        for attri in self.categorical_data[start_index:]:
            i += 1
            if len(set(attri)) > 500:
                continue
            categorical_name_list.append(self.name_index['categorical'][i])
            tokenized_data, tokenizer, tokenizer_re = tokenize_categorical_values(attri)
            generated_data.append(np.asarray(tokenized_data))
            tokenizers.append(tokenizer)
            tokenizer_reverses.append(tokenizer_re)
        self.name_index['categorical'] = categorical_name_list
        self.tokenizer_list = tokenizers
        self.reverse_tokenizer_dict = dict(zip(categorical_name_list, tokenizer_reverses))
        categorical_data_tokenized = np.stack(generated_data, axis=1)
        if self.date_flag:
            self.categorical_data = np.concatenate(
                [np.reshape(np.asarray(self.categorical_data[0]), newshape=(len(self.categorical_data[0]), 1)),
                 np.reshape(np.asarray(self.categorical_data[1]), newshape=(len(self.categorical_data[1]), 1)),
                 categorical_data_tokenized], axis=1)
        else:
            self.categorical_data = np.concatenate(
                [np.reshape(np.asarray(self.categorical_data[0]), newshape=(len(self.categorical_data[0]), 1)),
                 categorical_data_tokenized], axis=1)

        if self.normalization_method == 'standard':
            normalized_numerical_data, self.normalization_a, self.normalization_b = stand_normalization(
                np.asarray(self.numerical_data[:, start_index:], dtype=np.float),
                axis=0)
        elif self.normalization_method == 'uniform':
            normalized_numerical_data, self.normalization_a, self.normalization_b = uniform_normalization(
                np.asarray(self.numerical_data[:, start_index:], dtype=np.float),
                axis=0)
        else:
            normalized_numerical_data, self.normalization_a, self.normalization_b = uniform_normalization(
                np.asarray(self.numerical_data[:, start_index:], dtype=np.float),
                axis=0)
        normalized_numerical_data = np.asarray(normalized_numerical_data, np.float)
        normalized_numerical_data[np.isnan(normalized_numerical_data)] = 0.0
        self.numerical_data[:, start_index:] = normalized_numerical_data

        if self.date_flag:
            self.cat_dyn_index, self.cat_sta_index = get_splited_index(self.categorical_data, 'categorical',
                                                                       diffs_th=self.threhold_diffs,
                                                                       user_th=self.threhold_user)
            self.num_dyn_index, self.num_sta_index = get_splited_index(self.numerical_data, 'numerical',
                                                                       diffs_th=self.threhold_diffs,
                                                                       user_th=self.threhold_user)
        else:
            self.cat_sta_index = [True] * (self.categorical_data.shape[-1] - 1)
            self.num_sta_index = [True] * (self.numerical_data.shape[-1] - 1)
            self.cat_dyn_index = []
            self.num_dyn_index = []

    def append_data(self, file_path, reset=False):
        dump_file = self.generate_dump_file()
        latest_records = self.latest_records

        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            try:
                raise Exception('File type wrong')
            except Exception:
                print('The program cannot solve the file type of {}'.format(file_path.split('.')[-1]))
        data = data.head(100)
        columns_index = data.columns.values.tolist()

        numerical_data = list()
        categorical_data = list()

        if self.label_flag:
            for label_name in self.label_indicator:
                if label_name not in columns_index:
                    try:
                        raise Exception('Keys not found!')
                    except Exception:
                        print("label indicator '{}' is not in the data head list".format(self.label_indicator))
            labels = list()
            for label_name in self.label_indicator:
                columns = data.loc[:, label_name].get_values()
                unique_c = np.unique(columns)
                value_dict = self.label_value_dict[label_name]
                for v in unique_c:
                    if v not in value_dict:
                        value_dict[v] = len(value_dict)
                columns = [value_dict[v] for v in columns]
                labels.append(columns)
            data = data.drop(self.label_indicator, axis=1)
            labels = np.stack(labels, axis=1)
            self.labels = labels
        else:
            self.labels = []

        if self.userid_flag:
            if self.user_id_indicator not in columns_index:
                try:
                    raise Exception('Keys not found!')
                except Exception:
                    print("user ids indicator '{}' is not in the data head list".format(self.label_indicator))
            user_ids = data.loc[:, self.user_id_indicator].apply(hash).get_values()
            data = data.drop(self.user_id_indicator, axis=1)
        else:
            user_ids = np.asarray(range(len(data)))
        categorical_data.append(user_ids)
        numerical_data.append(user_ids)

        if self.date_flag:
            if self.date_indicator not in columns_index:
                try:
                    raise Exception('Keys not found!')
                except Exception:
                    print("indicator for date '{}' is not in the data head list".format(self.date_indicator))
            try:
                date_column = (pd.to_datetime(data.loc[:, self.date_indicator]).get_values() - np.datetime64(
                    '1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                data = data.drop(self.date_indicator, axis=1)
                categorical_data.append(date_column)
                numerical_data.append(date_column)
            except Exception:
                print("column named '{}' is not in date type".format(self.date_indicator))

        columns_index = data.columns.values.tolist()
        for i in range(len(columns_index)):
            if columns_index[i] in self.name_index['categorical']:
                categorical_data.append(data.iloc[:, i].get_values().astype('str'))
            elif columns_index[i] in self.name_index['numerical']:
                numerical_data.append(np.asarray(data.iloc[:, i].get_values()))
            else:
                try:
                    raise Exception('New features found!')
                except Exception:
                    print("Feature named {} is not included in previous data".format(columns_index[i]))
        numerical_data = np.stack(numerical_data, axis=1)
        if self.date_flag:
            start_index = 2
        else:
            start_index = 1
        i = 0
        generated_data = list()
        for attri in categorical_data[start_index:]:
            uni_a = np.unique(attri)
            for va in uni_a:
                if va not in self.tokenizer_list[i]:
                    self.tokenizer_list[i][va] = len(self.tokenizer_list[i])
                    self.reverse_tokenizer_dict[self.name_index['categorical'][i]][len(self.tokenizer_list[i])] = va
            tokenized_data, _, _ = tokenize_categorical_values(attri, self.tokenizer_list[i])
            generated_data.append(np.asarray(tokenized_data))
            i += 1
        categorical_data_tokenized = np.stack(generated_data, axis=1)
        if self.date_flag:
            categorical_data = np.concatenate(
                [np.reshape(np.asarray(categorical_data[0]), newshape=(len(categorical_data[0]), 1)),
                 np.reshape(np.asarray(categorical_data[1]), newshape=(len(categorical_data[1]), 1)),
                 categorical_data_tokenized], axis=1)
        else:
            categorical_data = np.concatenate(
                [np.reshape(np.asarray(categorical_data[0]), newshape=(len(categorical_data[0]), 1)),
                 categorical_data_tokenized], axis=1)

        if self.normalization_method == 'standard':
            normalized_numerical_data, self.normalization_a, self.normalization_b = stand_normalization(
                np.asarray(numerical_data[:, start_index:], dtype=np.float), means=self.normalization_a, stds=self.normalization_b,
                axis=0)
        elif self.normalization_method == 'uniform':
            normalized_numerical_data, self.normalization_a, self.normalization_b = uniform_normalization(
                np.asarray(numerical_data[:, start_index:], dtype=np.float), maxs=self.normalization_a, mins=self.normalization_b,
                axis=0)
        else:
            normalized_numerical_data, self.normalization_a, self.normalization_b = uniform_normalization(
                np.asarray(numerical_data[:, start_index:], dtype=np.float), maxs=self.normalization_a, mins=self.normalization_b,
                axis=0)
        normalized_numerical_data = np.asarray(normalized_numerical_data, np.float)
        normalized_numerical_data[np.isnan(normalized_numerical_data)] = 0.0
        numerical_data[:, start_index:] = normalized_numerical_data

        cat_dynamics_length = int(np.sum(self.cat_dyn_index))
        cat_statics_length = int(np.sum(self.cat_sta_index))
        static_records = []
        new_labels = []
        results = dict()
        results['seq_len'] = self.seq_len
        if self.date_flag:
            dynamic_records = []
            dynamic_data = np.concatenate([categorical_data[:, 0:start_index],
                                           categorical_data[:, start_index:][:, self.cat_dyn_index],
                                           numerical_data[:, start_index:][:, self.num_dyn_index]], axis=1)
            static_data = np.concatenate([categorical_data[:, 0:start_index],
                                          categorical_data[:, start_index:][:, self.cat_sta_index],
                                          numerical_data[:, start_index:][:, self.num_sta_index],
                                          labels], axis=1)
            statics_group = dict(
                [(user, np.asarray(sorted(list(records), key=lambda x: x[1]))) for user, records in groupby(static_data, lambda x: x[0])])
            if self.label_flag:
                label_num = labels.shape[-1]
            else:
                label_num = 0
            static_len = static_data.shape[-1]
            for user, records in groupby(dynamic_data, lambda x: x[0]):
                records = np.asarray(sorted(list(records), key=lambda x: x[1]))
                if len(records) < 1:
                    continue
                if user in self.latest_records:
                    new_record = np.concatenate([self.latest_records[user], records], axis=0)
                    start = max(0, len(new_record) - self.seq_len + 1)
                    self.latest_records[user] = new_record[start:, ]
                for i in range(0, len(records)):
                    seq = records[max(0, i - self.seq_len + 1):i + 1, start_index:]
                    if user in self.latest_records:
                        lenth = self.seq_len - len(seq)
                        seq = np.concatenate([self.latest_records[user][-lenth:,start_index:], seq], axis=0)
                    if len(seq) < self.seq_len:
                        seq = np.pad(seq, ((0, self.seq_len - seq.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    dynamic_records.append(seq)
                    static_records.append(statics_group[user][i][start_index:static_len - label_num])
                    new_labels.append(statics_group[user][i][static_len - label_num:])
                    
            dynamics = np.stack(dynamic_records, axis=0)
            print(dynamics.shape)
            statics = np.stack(static_records, axis=0)
            results['static_categorical_data'] = statics[:, :cat_statics_length]
            results['static_numerical_data'] = statics[:, cat_statics_length:]
            results['dynamic_categorical_data'] = dynamics[:, :, :cat_dynamics_length].transpose([2, 0, 1])
            results['dynamic_numerical_data'] = dynamics[:, :, cat_dynamics_length:]
            results['dynamic_categorical_token'] = get_columns(self.tokenizer_list, self.cat_dyn_index)
            results['static_categorical_token'] = get_columns(self.tokenizer_list, self.cat_sta_index)
        else:
            results['static_categorical_data'] = self.categorical_data[:, start_index:][:, self.cat_sta_index]
            results['static_numerical_data'] = self.numerical_data[:, start_index:][:, self.num_sta_index]
            results['static_categorical_token'] = self.tokenizer_list
            results['dynamic_numerical_data'] = []
            results['dynamic_categorical_data'] = np.asarray([[[]]])
            results['dynamic_numerical_data'] = np.asarray([[[]]])

        if self.label_flag:
            if not self.date_flag:
                results['labels'] = np.asarray(self.labels)
            else:
                results['labels'] = np.asarray(new_labels)
            results['label_name_dict'] = self.label_name_dict
        else:
            results['labels'] = []
            results['label_name_dict'] = []

        pickle.dump({'name_index': self.name_index,
                     'normalization_method': self.normalization_method,
                     'normalization_a': self.normalization_a,
                     'normalization_b': self.normalization_b,
                     'reverse_tokenizer_dict': self.reverse_tokenizer_dict,
                     'tokenizer_list': self.tokenizer_list,
                     'cat_dyn_index': self.cat_dyn_index,
                     'cat_sta_index': self.cat_sta_index,
                     'num_dyn_index': self.num_dyn_index,
                     'num_sta_index': self.num_sta_index,
                     'latest_records': latest_records,
                     'label_value_dict': self.label_value_dict},
                    open(dump_file + '.interp', "wb"))
        return results, dump_file + '.interp'

    def process_data(self):
        dump_file = self.generate_dump_file()
        if os.path.exists(dump_file + '.training'):
            data_file = dump_file + '.training'
            interpretation_file = dump_file + '.interp'
            interp_dict = pickle.load(open(interpretation_file, "rb"))
            self.name_index = interp_dict['name_index']
            self.normalization_method = interp_dict['normalization_method']
            self.normalization_a = interp_dict['normalization_a']
            self.normalization_b = interp_dict['normalization_b']
            self.reverse_tokenizer_dict = interp_dict['reverse_tokenizer_dict']
            self.tokenizer_list = interp_dict['tokenizer_list']
            self.cat_dyn_index = interp_dict['cat_dyn_index']
            self.cat_sta_index = interp_dict['cat_sta_index']
            self.num_dyn_index = interp_dict['num_dyn_index']
            self.num_sta_index = interp_dict['num_sta_index']
            self.label_value_dict = interp_dict['label_value_dict']
            self.label_name_dict = interp_dict['label_name_dict']
            return pickle.load(open(data_file, "rb")), dump_file + '.interp'
        self._read_file()
        self._data_clean()
        cat_dynamics_length = int (np.sum(self.cat_dyn_index))
        cat_statics_length = int (np.sum(self.cat_sta_index))

        static_records = []
        new_labels = []

        if self.date_flag:
            start_index = 2
        else:
            start_index = 1

        results = dict()
        results['seq_len'] = self.seq_len
        if self.date_flag:
            dynamic_records = []
            dynamic_data = np.concatenate([self.categorical_data[:, 0:start_index],
                                           self.categorical_data[:, start_index:][:, self.cat_dyn_index],
                                           self.numerical_data[:, start_index:][:, self.num_dyn_index]], axis=1)
            static_data = np.concatenate([self.categorical_data[:, 0:start_index],
                                          self.categorical_data[:, start_index:][:, self.cat_sta_index],
                                          self.numerical_data[:, start_index:][:, self.num_sta_index],
                                          self.labels], axis=1)
            statics_group = dict(
                [(user, np.asarray(list(records))) for user, records in groupby(static_data, lambda x: x[0])])
            if self.label_flag:
                label_num = self.labels.shape[-1]
            else:
                label_num = 0
            static_len = static_data.shape[-1]

            for user, records in groupby(dynamic_data, lambda x: x[0]):
                records = np.asarray(list(records))
                if len(records) < 1:
                    continue
                start = max(0, len(records) - self.seq_len + 1)
                self.latest_records[user] = records[start:, ]
                for i in range(0, max(1, len(records) - self.seq_len + 1)):
                    seq = records[i:min(len(records), i + self.seq_len), start_index:]
                    if len(records) < self.seq_len:
                        seq = np.pad(seq, ((0, self.seq_len - seq.shape[0]), (0, 0)), 'constant',
                                     constant_values=(0, 0))
                    dynamic_records.append(seq)
                    static_records.append(
                        statics_group[user][min(len(statics_group[user]), i + self.seq_len) - 1][start_index:static_len-label_num])
                    new_labels.append(
                        statics_group[user][min(len(statics_group[user]), i + self.seq_len) - 1][static_len-label_num:])
            dynamics = np.stack(dynamic_records, axis=0)
            statics = np.stack(static_records, axis=0)
            results['static_categorical_data'] = statics[:, :cat_statics_length]
            results['static_numerical_data'] = statics[:, cat_statics_length:]
            results['dynamic_categorical_data'] = dynamics[:, :, :cat_dynamics_length].transpose([2, 0, 1])
            results['dynamic_numerical_data'] = dynamics[:, :, cat_dynamics_length:]
            results['dynamic_categorical_token'] = get_columns(self.tokenizer_list, self.cat_dyn_index)
            results['static_categorical_token'] = get_columns(self.tokenizer_list, self.cat_sta_index)
        else:
            results['static_categorical_data'] = self.categorical_data[:, start_index:][:, self.cat_sta_index]
            results['static_numerical_data'] = self.numerical_data[:, start_index:][:, self.num_sta_index]
            results['static_categorical_token'] = self.tokenizer_list
            results['dynamic_numerical_data'] = []
            results['dynamic_categorical_data'] = np.asarray([[[]]])
            results['dynamic_numerical_data'] = np.asarray([[[]]])

        if self.label_flag:
            if not self.date_flag:
                results['labels'] = np.asarray(self.labels)
            else:
                results['labels'] = np.asarray(new_labels)
            results['label_dict'] = self.label_name_dict
        else:
            results['labels'] = []
            results['label_dict'] = []

        pickle.dump(results, open(dump_file + '.training', "wb"))

        pickle.dump({'name_index': self.name_index,
                     'normalization_method': self.normalization_method,
                     'normalization_a': self.normalization_a,
                     'normalization_b': self.normalization_b,
                     'reverse_tokenizer_dict': self.reverse_tokenizer_dict,
                     'tokenizer_list': self.tokenizer_list,
                     'cat_dyn_index': self.cat_dyn_index,
                     'cat_sta_index': self.cat_sta_index,
                     'num_dyn_index': self.num_dyn_index,
                     'num_sta_index': self.num_sta_index,
                     'latest_records': self.latest_records,
                     'label_value_dict': self.label_value_dict,
                     'label_name_dict': self.label_name_dict},
                    open(dump_file + '.interp', "wb"))
        return results, dump_file + '.interp'

def get_columns(alist, index_indicator):
    newlist = list()
    for i in range(len(index_indicator)):
        if index_indicator[i]:
            newlist.append(alist[i])
    return newlist
