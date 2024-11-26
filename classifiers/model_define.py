# build module
import logging
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b * doc_len * hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x doc_len

        # linear
        # key： b * doc_len * hidden
        key = torch.matmul(batch_hidden, self.weight) + self.bias

        # compute attention
        # matmul 会进行广播
        #outputs: b * doc_len
        outputs = torch.matmul(key, self.query)
        # 1 - batch_masks 就是取反，把没有单词的句子置为 0
        # masked_fill 的作用是 在 为 1 的地方替换为 value: float(-1e32)
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
        #attn_scores：b * doc_len
        attn_scores = F.softmax(masked_outputs, dim=1)

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        # masked_attn_scores.unsqueeze(1)：# b * 1 * doc_len
        # key：b * doc_len * hidden
        # batch_outputs：b * hidden
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)

        return batch_outputs, attn_scores
    

# 读取训练好的词向量文件
word2vec_path = './emb/word2vec.txt'
dropout = 0.15

# 输入是：
# 输出是：
class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100 # 词向量的长度是 100 维
        # padding_idx 表示当取第 0 个词时，向量全为 0
        # 这个 Embedding 层是可学习的
        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        # # 这个 Embedding 层是不可学习的
        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
        # 3 个卷积层，卷积核大小分别为 [2,100], [3,100], [4,100]
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, word_ids, extword_ids):
        # word_ids: sentence_num * sentence_len
        # extword_ids: sentence_num * sentence_len
        # batch_masks: `sentence_num` * sentence_len
        sen_num, sent_len = word_ids.shape

        # word_embed: sentence_num * sentence_len * 100
        # 根据 index 取出词向量
        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)
        # batch_embed: sentence_num x 1 x sentence_len x 100
        # squeeze 是为了添加一个 channel 的维度，成为 B * C * H * W
        # 方便下面做 卷积
        batch_embed.unsqueeze_(1)

        pooled_outputs = []
        # 通过 3 个卷积核做 3 次卷积核池化
        for i in range(len(self.filter_sizes)):
            # 通过池化公式计算池化后的高度: o = (i-k)/s+1
            # 其中 o 表示输出的长度
            # k 表示卷积核大小
            # s 表示步长，这里为 1
            filter_height = sent_len - self.filter_sizes[i] + 1
            # conv：sentence_num * out_channel * filter_height * 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)
            # 定义池化层
            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            # pooled：sentence_num * out_channel * 1 * 1 -> sen_num * out_channel
            # 也可以通过 squeeze 来删除无用的维度
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)

            pooled_outputs.append(pooled)
        # 拼接 3 个池化后的向量
        # reps: sen_num * (3*out_channel)
        reps = torch.cat(pooled_outputs, dim=1)

        if self.training:
            reps = self.dropout(reps)

        return reps
    
# build sent encoder
sent_hidden_size = 256
sent_num_layers = 2


class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size, # 每个句子经过 CNN 后得到 300 维向量
            hidden_size=sent_hidden_size,# 输出的维度
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b * doc_len * sent_rep_size
        # sent_masks: b * doc_len
        # sent_hiddens:  b * doc_len * hidden*2
        # sent_hiddens:  batch, seq_len, num_directions * hidden_size
        sent_hiddens, _ = self.sent_lstm(sent_reps)
        # 对应相乘，用到广播，是为了只保留有句子的位置的数值
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens
    

# build model
class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 300 # 经过 CNN 后得到的 300 维向量
        self.doc_rep_size = sent_hidden_size * 2 # lstm 最后输出的向量长度
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab)

        # set cuda
        gpu = 0
        self.use_cuda = gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(gpu)
            device = torch.device("cuda", gpu)
        else:
            device = torch.device("cpu")
        logging.info("Use cuda: %s, gpu id: %d.", self.use_cuda, gpu)

        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        # doc_rep_size
        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if self.use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        logging.info('Build model with cnn word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))
    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b * doc_len * sentence_len
        # batch_masks : b * doc_len * sentence_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        # batch_inputs1: sentence_num * sentence_len
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)
        # batch_inputs2: sentence_num * sentence_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
        # batch_masks: sentence_num * sentence_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)
        # sent_reps: sentence_num * sentence_rep_size
        # sen_num * (3*out_channel) =  sen_num * 300
        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)


        # sent_reps：b * doc_len * sent_rep_size
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)
        # batch_masks：b * doc_len * max_sent_len
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)
        # sent_masks：b * doc_len any(2) 表示在 第二个维度上判断
        # 表示如果如果一个句子中有词 true，那么这个句子就是 true，用于给 lstm 过滤
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len
        # sent_hiddens: b * doc_len * num_directions * hidden_size
        # sent_hiddens:  batch, seq_len, 2 * hidden_size
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)


        # doc_reps: b * (2 * hidden_size)
        # atten_scores: b * doc_len
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)

        # b * num_labels
        batch_outputs = self.out(doc_reps)

        return batch_outputs

# build optimizer

class Optimizer:
    def __init__(self, model_parameters,learning_rate = 2e-4,
                 decay = .75,decay_step = 1000):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)

                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res

# some function
from sklearn.metrics import f1_score, precision_score, recall_score
from Data_processing import get_examples, data_iter,split_data
import numpy as np


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)

# 保留 n 位小数点
def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))

# build trainer

import time
from sklearn.metrics import classification_report

clip = 5.0
epochs = 10
early_stops = 3
log_interval = 50

test_batch_size = 128
train_batch_size = 128
train_num = 200000

save_model = './cnn.bin'
save_test = './cnn.csv'

class Trainer():
    def __init__(self, model, vocab):
        self.model = model
        self.report = True

        gpu = 0
        self.use_cuda = gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(gpu)
            self.device = torch.device("cuda", gpu)
        else:
            self.device = torch.device("cpu")

        # get_examples() 返回的结果是 一个 list
        # 每个元素是一个 tuple: (label, 句子数量，doc)
        # 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
        train_data, dev_data, test_data = split_data(10,1,train_num)
        self.train_data = get_examples(train_data, vocab)
        logging.info('train_data loaded successfully.')
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data, vocab)
        logging.info('dev_data loaded successfully.')
        self.test_data = get_examples(test_data, vocab)
        logging.info('test_data loaded successfully.')

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = vocab.target_names

        # optimizer
        self.optimizer = Optimizer(model.all_parameters)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1 = self._eval(epoch)

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            # batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
            # 形状都是：batch_size * doc_len * sent_len
            # batch_labels: batch_size
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            # batch_outputs：b * num_labels
            batch_outputs = self.model(batch_inputs)
            # criterion 是 CrossEntropyLoss，真实标签的形状是：N
            # 预测标签的形状是：(N,C)
            loss = self.criterion(batch_outputs, batch_labels)

            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value
            # 把预测值转换为一维，方便下面做 classification_report，计算 f1
            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat 保留 4 位数字
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses, during_time))
        # 如果预测和真实的标签都包含相同的类别数目，才能调用 classification_report
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    # 这里验证集、测试集都使用这个函数，通过 test 来区分使用哪个数据集
    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                            # batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
            # 形状都是：batch_size * doc_len * sent_len
            # batch_labels: batch_size
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                # batch_outputs：b * num_labels
                batch_outputs = self.model(batch_inputs)
                # 把预测值转换为一维，方便下面做 classification_report，计算 f1
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time

            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info(
                    '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
                                                                              during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)

        return f1


    # data 参数就是 get_examples() 得到的，经过了分 batch
    # batch_data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
    # 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            # doc_data 代表一篇新闻，是一个 tuple: (label, 句子数量，doc)
            # doc_data[0] 是 label
            doc_labels.append(doc_data[0])
            # doc_data[1] 是 这篇文章的句子数量
            doc_lens.append(doc_data[1])
            # doc_data[2] 是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
            # 所以 sent_data[0] 表示每个句子的长度（单词个数）
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            # 取出这篇新闻中最长的句子长度（单词个数）
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        # 取出最长的句子数量
        max_doc_len = max(doc_lens)
        # 取出这批 batch 数据中最长的句子长度（单词个数）
        max_sent_len = max(doc_max_sent_len)
        # 创建 数据
        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                # batch_data[b][2] 表示一个 list，是一篇文章中的句子
                sent_data = batch_data[b][2][sent_idx] #sent_data 表示一个句子
                for word_idx in range(sent_data[0]): # sent_data[0] 是句子长度(单词数量)
                    # sent_data[1] 表示 word_ids
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    # # sent_data[2] 表示 extword_ids
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    # mask 表示 哪个位置是有词，后面计算 attention 时，没有词的地方会被置为 0
                    batch_masks[b, sent_idx, word_idx] = 1

        if self.use_cuda:
            batch_inputs1 = batch_inputs1.to(self.device)
            batch_inputs2 = batch_inputs2.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_labels = batch_labels.to(self.device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels

