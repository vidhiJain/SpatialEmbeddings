'''
Assumptions:
    word2vec vectors is downloaded into './data/GoogleNews-vectors-negative300.bin'
'''

import os
import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
import fasttext
import fasttext.util
from gensim.models import KeyedVectors
import networkx as nx

class Embeddings:
    def __init__(self):
        pass

    def change_case(self, str):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)

    def get_similarity(self, obj1, obj2):
        raise NotImplementedError

    def get_word_vector(self, obj1):
        raise NotImplementedError


class Word2VecEmbedding(Embeddings):
    """Word2Vec on GoogleNews"""
    def __init__(self):
        super(NoobEmbedding, self).__init__()
        self.model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        return 

    def get_similarity(self, obj1, obj2):
        obj1 = self.change_case(obj1)
        obj2 = self.change_case(obj2)
        sim = 0
        try:
            sim = self.model.similarity(obj1, obj2)
            return sim
        except Exception as e:
            print(e)
            return 0

    def get_word_vector(self, obj1):
        return self.model.get_vector(obj1)


class FastTextEmbedding(Embeddings):
    """FastText embeddings"""
    def __init__(self):
        super(FastTextEmbedding, self).__init__()
        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.model = fasttext.load_model('cc.en.300.bin')
        return

    def get_similarity(self, obj1, obj2):
        obj1 = self.change_case(obj1)
        obj2 = self.change_case(obj2)
        sim = 0
        try:
            emb1 = self.model.get_word_vector(obj1)
            emb2 = self.model.get_word_vector(obj2)
        except Exception as e:
            print(e)
            return 0
        assert len(emb1) == len(emb2)
        sim = cos_sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        if not np.isnan(sim):
            return sim
        else:
            return 0

    def get_nearest_neighbors(self, obj1):
        obj1 = self.change_case(obj1)
        nearest_neighbors = self.model.get_nearest_neighbors(obj1)
        return nearest_neighbors

    def get_word_vector(self, obj1):
        return self.model.get_word_vector(obj1)


class RobocseEmbedding(Embeddings):
    """Robocse ent embeddings"""
    def __init__(self):
        super(RobocseEmbedding, self).__init__()
        self.mapping = pd.read_csv("./data/entity2id.txt", header=None, skiprows=[0], sep = '\t').to_numpy()
        self.embeddings = np.load("./data/robocse_vectors.npy")

    def get_word_vector(self, obj1):
        obj1 = self.change_case(obj1)
        # Handle cases where object is not present, this should not happen 
        if not obj1 in self.mapping[:,0]:
            return np.zeros((1, self.embeddings.shape[-1]))
        return self.embeddings[self.mapping[:,0]==obj1]

    def change_case(self, obj1):
        # Making it compatible with the ROBOCSE mapping
        if obj1[-2] != ".":
            obj1 = obj1.lower() + '.o'
        return obj1

    def get_similarity(self, obj1, obj2):
        sim = 0
        try:
            emb1 = np.squeeze(self.get_word_vector(obj1))
            emb2 = np.squeeze(self.get_word_vector(obj2))
            emb2_l = np.squeeze(self.get_word_vector(self.change_case(obj2)[:-1]+"l"))
        except Exception as e:
            print(e)
            return 0
        assert len(emb1) == len(emb2)
        sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        sim_l = np.dot(emb1, emb2_l)/(np.linalg.norm(emb1)*np.linalg.norm(emb2_l))
        if np.isnan(sim):
            sim = -2
        if np.isnan(sim_l):
            sim_l = -2
        return max(sim, sim_l)


class GraphEmbedding(RobocseEmbedding):
    """Graph ent embeddings"""
    def __init__(self):
        super(GraphEmbedding, self).__init__()
        self.mapping = pd.read_csv("./data/graph_columns.tsv", header=None, sep = '\t').to_numpy()
        self.mapping = np.squeeze(self.mapping)
        self.embeddings = pd.read_csv("./data/graph_vectors.tsv", header=None, sep = '\t').to_numpy()

    def get_word_vector(self, obj1):
        obj1 = self.change_case(obj1)
        # Handle cases where object is not present, this should not happen 
        if not obj1 in self.mapping[:]:
            return np.zeros((1, self.embeddings.shape[-1]))
        return self.embeddings[self.mapping[:]==obj1]

