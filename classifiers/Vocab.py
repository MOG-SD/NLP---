# build vocab
import logging
import Data_processing
import numpy as np
from collections import Counter
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer()

# Vocab 的作用是：
# 1. 创建 词 和 index 对应的字典，这里包括 2 份字典，分别是：_id2word 和 _id2extword
# 其中 _id2word 是从新闻得到的， 把词频小于 5 的词替换为了 UNK。对应到模型输入的 batch_inputs1。
# _id2extword 是从 word2vec.txt 中得到的，有 5976 个词。对应到模型输入的 batch_inputs2。
# 后面会有两个 embedding 层，其中 _id2word 对应的 embedding 是可学习的，_id2extword 对应的 embedding 是从文件中加载的，是固定的
# 2.创建 label 和 index 对应的字典

class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        #创建词和 index 对应的字典
        self._word2id = reverse(self._id2word)
        #创建 label 和 index 对应的字典
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    #创建词典
    def build_vocab(self, data):
        self.word_counter = Counter()
        #计算每个词出现的次数
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1
        # 去掉频次小于 min_count = 5 的词，把词存到 _id2word
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label] # 取出 label 对应的次数
            self._id2label.append(label)
            self.target_names.append(label2name[label]) # 根据label数字取出对应的名字

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            # 第一行分别是单词数量、词向量维度
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        # 下面的代码和 word2vec.txt 的结构有关
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0]) # 首先添加第一列的单词
            vector = np.array(values[1:], dtype='float64') # 然后添加后面 100 列的词向量
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        # unk 的词向量是所有词的平均
        embeddings[self.unk] = embeddings[self.unk] / word_count
        # 除以标准差干嘛？
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    # 根据单词得到 id
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)
    # 根据单词得到 ext id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)
    # 根据 label 得到 id
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)



