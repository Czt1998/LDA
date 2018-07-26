#coding=utf-8

import jieba  # 结巴分词工具
import jieba.posseg
import numpy as np
import lda
from Tools import *
from Learn_MyLDA import *
from Learn_MyCluster import *

def main(movie_id):
    file_path = "./data/seg/" + str(movie_id)
    list_likelihood = []  # 记录每次迭代的复杂度
    list_ste = [] # 记录每次迭代中的损失主题代价
    list_topic = []   # 记录每次迭代的主题词语
    list_number = [8,9,10,11,12,13,14,15,16]
    list_error = [] # 损失主题代价
    # list_number = [5,8,10,12]

    # 训练LDA
    for i in range(0,list_number.__len__(),1):
        n_topics = list_number[i]
        lda_model = LDAModel(file_path,n_topics=n_topics)
        perplxity = lda_model.get_perplexity()
        topics = lda_model.get_topic() # 获得词分布
        sum_topic_error = lda_model.count_ste()
        list_likelihood.append(perplxity)
        list_error.append(sum_topic_error)
        list_topic.append(topics)
        mycluster = MyCluster(topics,file_path)
        group = mycluster.get_group()
        valid_group = mycluster.count_valid_group()
        list_ste.append(valid_group)

    print(list_likelihood)
    print(list_ste)

    list_srm = list()
    max = -1
    index = -1
    for i in range(0,list_ste.__len__(),1):
        srm = list_ste[i]
        list_srm.append(srm)
        if srm > max:
            max = srm
            index = i

    # output best ci fen bu
    for i in topics:
        print('------- topic %s ----------' % i)
        for word in topics[i]:
            print(word)
    topic_file = "./data/topic/topic " + str(movie_id)
    write_topic(topics, topic_file)
    print("----------store----------")

def write_topic(topics, filename):
    lines = ''
    num = 0
    for i in topics:
        line = str(num) + ':'
        for word in topics[i]:
            line += word + " "
        lines += line.replace('\n', '').strip() + '\n'
        num += 1
    with open(filename,"w") as w:
        w.write(lines)

if __name__ == '__main__':
    MovieList=GetMoviesList()
    for movie in MovieList:
        try:
            Seg(movie)
        except Exception as e:
            continue
        main(movie)
