from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from collections import OrderedDict
# from models.aug_slot_dec import slot_descs as aug_dec
from models.aug_slot_dec import slot_descs as aug_dec
from numpy import random

slot_descs = aug_dec
# from transformers import BertTokenizer

def text_to_uri(text):
    """
    An extremely cut-down version of ConceptNet's `standardized_concept_uri`.
    Converts a term such as "apple" into its ConceptNet URI, "/c/en/apple".

    Only works for single English words, with no punctuation besides hyphens.
    """
    return '/c/en/' + text.lower().replace('-', '_')


def normalize_vec(vec):
    """
    Normalize a vector to a unit vector, so that dot products are cosine
    similarities.

    If it's the zero vector, leave it as is, so all its cosine similarities
    will be zero.
    """
    norm = np.abs(vec.dot(vec)) ** 0.5
    if norm == 0:
        return vec
    return vec / norm


class AttributeHeuristic:
    def __init__(self, hdf5_filename):
        """
        Load a word embedding matrix that is the 'mat' member of an HDF5 file,
        with UTF-8 labels for its rows.
        (This is the format that ConceptNet Numberbatch word embeddings use.)
        """
        self.embeddings = pd.read_hdf(hdf5_filename, 'mat', encoding='utf-8')
        self.cache = {}

    def get_vector(self, term):
        """
        Look up the vector for a term, returning it normalized to a unit vector.
        If the term is out-of-vocabulary, return a zero vector.

        Because many terms appear repeatedly in the data, cache the result.
        """
        uri = text_to_uri(term)
        if uri in self.cache:
            return self.cache[uri]  # 缓存命中，直接返回
        else:
            try:
                vec = normalize_vec(self.embeddings.loc[uri])  # 正则化
                # vec = self.embeddings.loc[uri]
            except KeyError:
                # vec = random.normal(size=(1, 300))
                vec = pd.Series(index=self.embeddings.columns, dtype='float64').fillna(np.random.normal(loc=0, scale=0.05))
            self.cache[uri] = vec.tolist()
            return vec.tolist()

    def get_sentence_vector(self, sentence):
        emb = []
        for word in sentence.split(' '):
            emb.append(self.get_vector(word))
        return emb

    def get_sentence_word_list(self, sentence):
        emb = []
        for word in sentence:
            emb.append(self.get_vector(word))
        return emb

    def get_similarity(self, term1, term2):
        """
        Get the cosine similarity between the embeddings of two terms.
        """
        t1v = self.get_vector(term1)
        t2v = self.get_vector(term2)
        denom = np.linalg.norm(t1v) * np.linalg.norm(t2v)
        return t1v.dot(t2v) / denom if denom != 0 else 0
        # return self.get_vector(term1).dot(self.get_vector(term2))

    def compare_attributes(self, term1, term2, attribute):
        """
        Our heuristic for whether an attribute applies more to term1 than
        to term2: find the cosine similarity of each term with the
        attribute, and take the difference of the square roots of those
        similarities.
        """
        match1 = max(0, self.get_similarity(term1, attribute)) ** 0.5
        match2 = max(0, self.get_similarity(term2, attribute)) ** 0.5
        return match1 - match2

    def classify(self, term1, term2, attribute, threshold):
        """
        Convert the attribute heuristic into a yes-or-no decision, by testing
        whether the difference is larger than a given threshold.
        """
        return self.compare_attributes(term1, term2, attribute) > threshold

    def evaluate(self, semeval_filename, threshold):
        """
        Evaluate the heuristic on a file containing instances of this form:

            banjo,harmonica,stations,0
            mushroom,onions,stem,1

        Return the macro-averaged F1 score. (As in the task, we use macro-
        averaged F1 instead of raw accuracy, to avoid being misled by
        imbalanced classes.)
        """
        our_answers = []
        real_answers = []
        for line in open(semeval_filename, encoding='utf-8'):
            term1, term2, attribute, strval = line.rstrip().split(',')
            discriminative = bool(int(strval))
            real_answers.append(discriminative)
            our_answers.append(self.classify(term1, term2, attribute, threshold))

        return f1_score(real_answers, our_answers, average='macro')


if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained('/data/lyh/mlm2')
    attribute = AttributeHeuristic('/home/fzus/lyh/DiCoS/models/mini.h5')
    knowledge_embedding = OrderedDict()
    for key in slot_descs.keys():
        knowledge_embedding[key] = attribute.get_sentence_vector(slot_descs[key])

    json_emb = json.dumps(knowledge_embedding)
    with open('/home/fzus/lyh/DiCoS/utils/slot_knowledge_emb.json', 'w') as json_file:
        json_file.write(json_emb)

    print(normalize_vec(torch.tensor([1, -1, -3, 2, 0, 1, -2, 0, 8, 3, 1, 5, -4, -4, -3, -5, -5, -4, -3, 4, 5, 4, 0, -2, 1, 2, 3, -2, -1, -4, -5, 5, -1, -1, 1, 0, 0, 2, -1, 0, 10, 0, -2, -2, -6, -2, 0, -4, -5, 0, 0, 0, 1, -1, 1, 0, 3, 3, -2, 0, -2, 2, -3, -2, 0, 0, -3, 3, 1, -6, 0, -3, 0, 2, 0, 6, -3, -3, -5, 1, -2, 1, 0, -6, -4, 2, 0, -1, 1, -1, 0, -4, 0, -4, 5, 0, -3, 0, 0, 6, 4, -9, -3, 0, -2, 5, 0, 5, -2, -2, 0, 6, 5, 6, -3, -3, -7, -5, 0, -1, 1, 0, -5, 1, 0, -2, 9, 0, 0, 0, -6, -1, 2, -2, 4, -3, 0, -1, 5, 0, 2, 0, 3, -7, -5, -2, -1, 0, 1, -9, 1, -2, -2, 0, 8, 5, 4, 1, 0, 8, 5, -1, -2, 2, -8, 0, 2, 1, 1, 9, 1, 0, 1, -3, 6, -4, -1, 0, 1, 8, -6, 1, 0, -1, -3, 12, 0, -5, 4, 0, 1, -1, 0, 1, -4, 4, 2, 4, 3, -1, 4, 0, -4, -2, 0, -1, 7, 2, 4, -5, 3, -2, 4, 1, -2, -6, 1, 4, -4, -4, -2, -3, -2, 2, 0, -4, 0, 2, 3, 0, 0, -2, 3, -4, 4, 1, -4, -3, 1, 0, 2, -1, -5, 2, 0, -2, 3, 0, 0, -1, 4, 0, 0, -3, -2, 0, 1, 3, -2, 0, -1, -4, -4, -3, 1, 0, 2, 3, 0, 0, -3, -5, -2, -2, -2, 4, -3, 0, -2, 0, 0, 6, 0, -7, 0, -5, 3, 0, 0, 4, 5, -3, -4, 6, 2, 2, 6, 1, 0, 0])))
