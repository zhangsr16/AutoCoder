import re
import joblib
class CustomTokenizer:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def tokenize(self, text):
        return self.pattern.findall(text)

# 定义自定义分词模式
pattern = r'\w+|[^\w\s]+'
tokenizer = CustomTokenizer(pattern)

# 示例文本数据
text_data = """自然语言处理是人工智能的一个重要领域。它涉及计算机科学、人工智能和语言学等多个领域。"""

tokens = tokenizer.tokenize(text_data)
print(tokens)

# 保存模型
joblib.dump(tokenizer, 'tokenizer.joblib')


# 加载模型
tokenizer = joblib.load('tokenizer.joblib')

# 测试分句器
sentences = tokenizer.tokenize(text_data)
print(sentences)

import jieba
import jieba.posseg as pseg
import jieba.analyse

text = "自然语言处理是人工智能的一个重要领域。"
tokens = jieba.lcut(text)
print("词元：", tokens)

# 词性标注
words = pseg.lcut(text)
print("词性标注：", [(word, flag) for word, flag in words])

# 命名实体识别（示例，jieba 不支持完整的 NER，可以使用其他库如 stanfordnlp）
# 这里使用词性标注来简单展示
named_entities = [word for word, flag in words if flag in ('nr', 'ns', 'nt', 'n')]
print("命名实体：", named_entities)

# 词元频次（使用 TF-IDF 作为示例）
tfidf = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
print("词元频次（TF-IDF）：", tfidf)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import jieba
from gensim.models import Word2Vec
import torch


# 获取文档的BERT嵌入
def get_bert_embedding(text, word_vectors):
    outputs = []
    TOKENS_MAXNUM = 10
    pos = 0
    for token in text:
        outputs.append(torch.tensor(word_vectors[token]))
        pos += 1
    for i in range(TOKENS_MAXNUM - pos):
        outputs.append(torch.zeros(word_vectors.vector_size))
    return torch.stack(tuple(outputs), dim=0)


# 示例文档和标签
documents = [
    "自然语言处理是人工智能的一个重要领域。",
    "机器学习和深度学习是人工智能的关键技术。",
    "我今天很高兴。",
    "这部电影非常糟糕。"
]
labels = [0, 0, 1, 1]  # 0: 科技类, 1: 情感类

# 分词
tokenized_documents = [jieba.lcut(doc) for doc in documents]

# 训练Word2Vec模型
model = Word2Vec(sentences=tokenized_documents, vector_size=len(tokenized_documents), window=len(tokenized_documents),
                 min_count=1, workers=4)

# 获取嵌入向量
word_vectors = model.wv
X = [get_bert_embedding(text, word_vectors) for text in tokenized_documents]
X = torch.stack(tuple(X), dim=0)
X = X.view(X.shape[0], -1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 训练逻辑回归分类器
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 预测和评估
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"分类准确率: {accuracy}")
