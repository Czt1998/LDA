# coding utf-8

class MyCluster():
    def __init__(self, topic_dict, document_path):
        self.theta = 0.2
        self.topics = self.get_topic(topic_dict)
        self.comment = self.load_file(document_path)
        self.group = [i * 0.0 for i in range(0, self.topics.__len__() + 1, 1)]

    # tiqu zhu ti ci
    def get_topic(self,topic_dict):
        list_topic = list()
        for i in topic_dict:
            list_topic.append(topic_dict[i])
        return list_topic

    def cluster(self):
        for comment in self.comment:
            comment_words = comment.split(" ")
            list_jaccard = [i*0.0 for i in range(0,self.topics.__len__(),1)]
            for i in range(0,self.topics.__len__(),1):
                topic_word = self.topics[i]
                jaccard = self.count_jaccard(comment_words,topic_word)
                list_jaccard[i] = jaccard
            max = 0
            index = -1
            for i in range(0,list_jaccard.__len__(),1):
                if list_jaccard[i] > max:
                    max = list_jaccard[i]
                    index = i
            if max < self.theta:
                self.group[self.topics.__len__()] += 1
            else:
                self.group[index] += 1
        return

    def get_group(self):
        return self.group

    def count_valid_group(self):
        number = 0
        for i in self.group:
            if i > 20:
                number += 1
        return number-1

    def count_jaccard(self,list_a,list_b):
        union = 0.0
        for i in list_a:
            if i in list_b:
                union += 1
        if union >= 4:
            union /= list_a.__len__() + list_b.__len__() - union
        else:
            union = 0
        return union

    @staticmethod
    def load_file(file_path):
        with open(file_path,"r") as r:
            lines = r.readlines()
            return lines


