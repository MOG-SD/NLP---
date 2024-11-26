import logging
import Vocab
import numpy as np
import random
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
# # set cuda
# gpu = 0
# use_cuda = gpu >= 0 and torch.cuda.is_available()
# if use_cuda:
#     torch.cuda.set_device(gpu)
#     device = torch.device("cuda", gpu)
# else:
#     device = torch.device("cpu")
# logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# split data to 10 fold
# fold_num = 10
import pandas as pd


def all_data2fold(fold_num, num=10000):
    data_file = './data/train_set.csv'
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)

    # print("total size", total)

    index = list(range(total))
    # 打乱数据
    np.random.shuffle(index)

    # all_texts 和 all_labels 都是 shuffle 之后的数据
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    # 构造一个 dict，key 为 label，value 是一个 list，存储的是该类对应的 index
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    # print("label2id", label2id.keys())

    # all_index 是一个 list，里面包括 10 个 list，称为 10 个 fold，存储 10 个 fold 对应的 index
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        # other 表示多出来的数据，other 的数据量是小于 fold_num 的
        other = len(data) - batch_size * fold_num
        # 把每一类对应的 index，添加到每个 fold 里面去
        for i in range(fold_num):
            # 如果 i < other，那么将一个数据添加到这一轮 batch 的数据中
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            # batch_data 是该轮 batch 对应的索引
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0

    # 由于上面在分 batch 的过程中，每个 batch 的数据量不一样，这里是把数据平均到每个 batch
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size: # 如果大于 batch_size 那么就取 batch_size 大小的数据
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size: # 如果小于 batch_size，那么就补全到 batch_size 的大小
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)
        # 这里是为了打乱数据
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data

# fold_data 是一个 list，有 10 个元素，每个元素是 dict，包括 label 和 text
# fold_data = all_data2fold(10)


# # build train, dev, test data
# fold_id = 9

# # dev
# dev_data = fold_data[fold_id]

# # train 取出前 9 个 fold 的数据
# train_texts = []
# train_labels = []
# for i in range(0, fold_id):
#     data = fold_data[i]
#     train_texts.extend(data['text'])
#     train_labels.extend(data['label'])

# train_data = {'label': train_labels, 'text': train_texts}

# # test 读取测试集数据
# test_data_file = './data/test_a.csv'
# f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
# texts = f['text'].tolist()
# test_data = {'label': [0] * len(texts), 'text': texts}

def split_data(fold_num,dev_num,num=10000):
    fold_id = fold_num - dev_num
    fold_data = all_data2fold(fold_num,num)
    dev_data = fold_data[fold_id]

    train_texts = []
    train_labels = []

    for i in range(0, fold_id):
        data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

    train_data = {'label': train_labels, 'text': train_texts}

    # test 读取测试集数据
    test_data_file = './data/test_a.csv'
    f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()
    test_data = {'label': [0] * len(texts), 'text': texts}
    return train_data, dev_data, test_data


# 作用是：根据一篇文章，把这篇文章分割成多个句子
# text 是一个新闻的文章
# vocab 是词典
# max_sent_len 表示每句话的长度
# max_segment 表示最多有几句话
# 最后返回的 segments 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):

    words = text.strip().split()
    document_len = len(words)
    # 划分句子的索引，句子长度为 max_sent_len
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        # 根据索引划分句子
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        # 把出现太少的词替换为 UNK
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        # 添加 tuple:(句子长度，句子本身)
        segments.append([len(segment), segment])

    assert len(segments) > 0
    # 如果大于 max_segment 句话，则局数减少一半，返回一半的句子
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        # 否则返回全部句子
        return segments

# 最后返回的数据是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # sents_words: 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            # 把 word 转为 id
            word_ids = vocab.word2id(sent_words)
            # 把 word 转为 ext id
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids]) # 句子长度，word_ids, extword_ids存疑
        examples.append([id, len(doc), doc])    # label, 句子数量，doc，存疑
        # 这里的doc和examples的元素 都是list，但是不知道为什么博主说是tuple

    logging.info('Total %d docs.' % len(examples))
    return examples


# build loader
# data 参数就是 get_examples() 得到的
# data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        # 如果 i < batch_num - 1，那么大小为 batch_size，否则就是最后一批数据
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


# data 参数就是 get_examples() 得到的
# data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        # 这里是打乱所有数据
        np.random.shuffle(data)
        # lengths 表示的是 每篇文章的句子数量
        lengths = [example[1] for example in data]
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data
    # 把 batch 的数据放进一个 list
    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        # 打乱 多个 batch
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch

