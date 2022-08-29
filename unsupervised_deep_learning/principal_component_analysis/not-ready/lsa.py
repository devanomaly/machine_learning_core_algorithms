# FIXME:>> ascii/utf-8... what's wrong in the encoding??? (error in tokenization!)
# we now use PCA to do Latent Semantic Analysis (LSA)
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open("../../data/nlp-stuff/all_book_titles.txt")]

stopwords = set(w.rstrip() for w in open("../../data/nlp-stuff/stopwords.txt"))
stopwords = stopwords.union(
    {
        "introduction",
        "edition",
        "series",
        "application",
        "approach",
        "card",
        "access",
        "package",
        "plus",
        "etext",
        "brief",
        "vol",
        "volume",
        "fundamental",
        "guide",
        "essential",
        "printed",
        "third",
        "second",
        "fourth",
    }
)


def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


word_idx_map = {}
current_idx = 0
all_tokens = []
all_titles = []
idx_word_map = []
for title in titles:
    # try:
    title = title.encode("ascii", "ignore")
    all_titles.append(title)
    tokens = my_tokenizer(title)
    all_tokens.append(tokens)
    for token in tokens:
        if token not in word_idx_map.keys():
            word_idx_map[token] = current_idx
            current_idx += 1
            idx_word_map.append(token)
    # except:
    #     print("exception...")
    #     pass


def tokens2vec(tokens):
    x = np.zeros(len(word_idx_map))
    for t in tokens:
        i = word_idx_map[t]
        x[i] = 1
    return x


N = len(all_tokens)
D = len(word_idx_map)
X = np.zeros((D, N))
i = 0

for tokens in all_tokens:
    print("i:", i)
    X[:, i] = tokens2vec(tokens)
    i += 1

svd = TruncatedSVD()
Z = svd.fit_transform(X)

plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=idx_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()
