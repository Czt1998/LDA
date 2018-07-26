# coding utf-8
import numpy as np
import lda
from Tools import *

class LDAModel:
    def __init__(self,document_path, n_topics = 10):
        # sentence_with_segment列表： 一行一行的单词，每一行构成一个单位
        # vocabulary列表： 单词列表，以单词为单位
        self.sentence_with_segment,self.vocabulary = self.load_document(document_path)
        self.vsm_model = self.train_vsm_model()
        self.n_iter = 300
        self.n_top_words = 10
        self.n_topics = n_topics
        self.words_in_topic, self.perplexity = self.train_lda_model(self.n_topics)  # 获得lda主题模型下的词分布

    @staticmethod
    def load_document(document_path):
        """
        对文件中的单词进行提取，处理为两个列表
        :param document_path: 文件路径
        :return: 词行列表sentence_with_segment，单词列表vocabulary
        """

        sentence_with_segment = list()
        vocabulary = list()
        # 打开seg文件夹
        with open(document_path, "r") as r:
            sentences = r.readlines()
            for sentence in sentences:
                # 获得由单词组成的列表
                list_word = sentence.split(" ")
                # print("-------list_word in load_document-------")
                # print(list_word)
                # print("----------------------------------------")
                # 将表格加进sentence_with_segment,以列表为单位
                sentence_with_segment.append(list_word)
                # 形成单词列表
                for word in list_word:
                    if vocabulary.count(word) == 0:
                        vocabulary.append(word)
        return sentence_with_segment, vocabulary

    def train_vsm_model(self):
        """
        利用sentence_with_segment和vocabulary训练vsm模型
        :return: vsm_model(矩阵)
        """

        vsm_model = list()
        for sentence in self.sentence_with_segment:
            # vsm为vsm_model的每一行数据
            # 首先设零
            vsm = [i*0 for i in range(0,self.vocabulary.__len__(),1)]
            # 若单词存在，则该行该单词位置+1
            for word in sentence:
                index = self.vocabulary.index(word)
                vsm[index] += 1
            vsm_model.append(vsm)
        # ？
        vsm_model = np.array(vsm_model)
        return vsm_model

    def train_lda_model(self,n_topics):
        """
        训练LDA模型
        :param topic_number: 主题的个数
        :return: words_in_topic 主题内的词分布
                ep : 0: 冬雨 大鹏 演技 倪妮 感觉 演员 喜欢 角色 尴尬
        """
        words_in_topic = dict() # 主题词的分布
        model = lda.LDA(n_topics=n_topics, n_iter=self.n_iter, random_state=1)
        # 填充vsm模型
        model.fit(self.vsm_model)
        # 主题词
        topic_word = model.topic_word_
        loglikelihood = model.loglikelihoods_
        # 计算复杂度
        perplexity = loglikelihood.pop() * (-1.0) / self.vocabulary.__len__() * self.n_topics
        n_top_words = self.n_top_words

        for i, topic_dict in enumerate(topic_word):
            # 符合该主题词的词语 ?
            topic_words = np.array(self.vocabulary)[np.argsort(topic_dict)][:-(n_top_words+1):-1]
            # 第i个主题词包含的词语
            words_in_topic[i] = topic_words
        return words_in_topic,perplexity

    # 得到主题分布
    def get_topic(self):
        return self.words_in_topic

    # 得到复杂度
    def get_perplexity(self):
        return self.perplexity

    # 计算误差主题代价
    def count_ste(self):
        sum_topic_error = 0.0
        # 获得主题中出现的词语
        topic_vocabulary = []
        for i in self.words_in_topic:  # 列举主题id（如1-10）
            for word in self.words_in_topic[i]:
                if word not in topic_vocabulary:
                    topic_vocabulary.append(word)  # 获得主题中出现的词语

        for i in self.words_in_topic:
            for j in self.words_in_topic:
                if i == j:
                    continue
            sum_topic_error += jaccard(self.words_in_topic[i], self.words_in_topic[j])
        print (1.0 * sum_topic_error / (self.words_in_topic.__len__() * 2))
        return 1.0 * sum_topic_error / (self.words_in_topic.__len__() * 2)

# ?
def jaccard(list_a, list_b):
    union = 0.0
    for i in list_a:
        if i in list_b:
            union += 1
    union /= list_a.__len__() + list_b.__len__() - union
    return union









