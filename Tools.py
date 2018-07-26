#coding=utf-8
import jieba

class WordSegemention:

    def __init__(self, sentence):
        self.sentence = sentence
        self.list_word = self.word_segment()

    def word_segment(self):

        """
        function:利用结巴分词工具分词
        :param sentence: 分词前的句原
        :return: list_word:分词后的词原
        """

        words = jieba.posseg.cut(self.sentence)
        list_word = list()
        for word in words:
            # https://blog.csdn.net/huludan/article/details/52727298
            # 筛选掉某些词性的词语
            if word.flag in ("zg","c","r","d","x","o","uj","ul","eng","m","m","p","y","q"):
                continue
            # 去除标点符号
            if word.word in ("。","，","：","；"):
                continue
            # 去除单字评论
            if len(word.word)<2:
                continue
            list_word.append(word.word)
        return list_word

    def get_words_with_segmention(self):
        return self.list_word


def Seg(cbo_id):
    wf = open("data/seg/"+str(cbo_id), "w")
    with open("data/comment/"+str(cbo_id)) as rf:
        line = rf.readline()
        while line:
            line = line.replace("\n", "")
            seq = WordSegemention(line).get_words_with_segmention()
            line = rf.readline()
            for word in seq:
                wf.write(word + " ")
            wf.write("\n")
            # print a.get_words_with_segmention()[0]

# 获取电影名称
def GetMoviesList():
    list=[]
    with open("2017Movies.csv",'r') as rf:
        line=rf.readline()
        while line:
            list.append(int(line.split(',')[0]))
            line=rf.readline()
    print (list)
    return list
