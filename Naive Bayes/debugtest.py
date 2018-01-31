#!/usr/bin/env python
# -*- coding: utf-8 -*-


def textParse(bigString):
    import re
    listOfTokens = re.split('\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 创建一个包含所有文档中出现的不重复词的列表
def createVocabList(dataSet ):
    vocabSet = set ([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
        # print vocabSet
    return list(vocabSet)


# （词袋模型）输入词汇表及某个文档，输出文档向量，向量的每个元素为1或0，分别表示词汇中的单词再输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word：%s is not in my Vocabulary!" % word )
    return returnVec


# 将垃圾邮件以及正常邮件进行分词、构造词袋模型、分割训练以及测试集
def spamTest():
    import random
    import numpy as np
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open("/Users/zhangxingbin/ml_algorithms/data/email/spam/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/Users/zhangxingbin/ml_algorithms/data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    return np.asarray(trainMat), np.asarray(trainClasses), np.asarray(testSet)


if __name__ == "__main__":
    spamTest()
