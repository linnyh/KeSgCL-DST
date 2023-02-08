import json
import re
import time

import sys
sys.path.append('')

import pandas as pd
from tqdm import tqdm
import functools
import torch
from models.ConceptNet import *
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import matplotlib as mpl
import torch.nn.functional as F



def get_matrix_triad(coo_matrix, data=False):
    '''
	获取矩阵的元组表示  (row,col)
	data 为 True 时 (row,col,data)
	:dependent  scipy
	:param coo_matrix: 三元组表示的稀疏矩阵  类型可以为 numpy.ndarray
	:param data: 是否需要 data值
	:return: list
	'''
    coo_matrix = coo_matrix.numpy()
    # 检查类型
    if not sp.isspmatrix_coo(coo_matrix):
        # 转化为三元组表示的稀疏矩阵
        coo_matrix = sp.coo_matrix(coo_matrix)
    # nx3的矩阵  列分别为 矩阵行，矩阵列及对应的矩阵值
    temp = np.vstack((coo_matrix.row, coo_matrix.col, coo_matrix.data)).transpose()
    return temp.tolist()


def search(word, data, n=1):
    result = data[data['start'].str.contains(word) & (
            data['relation'].str.contains('Synonym') | data['relation'].str.contains('IsA'))]
    topK_result = result.sort_values("weights", ascending=False).head(n)
    children = []
    # 转成Node对象列表
    for idx, row in topK_result.iterrows():
        print("当前找到 [{}] 的子节点 [{}]".format(word, strip_end(row[3])))
        child = Node(strip_end(row[3]))
        children.append(child)
    return children


def strip_start(str):
    return str.split('/')[3]


def strip_end(str):
    return str.split('/')[3]


def strip_relation(str):
    return str.split('/')[2]


class Node:
    def __init__(self, word):
        self.start = word
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def set_children(self, children):
        self.children = children

    def get_children(self):
        return self.children


# 查找实体
def get_entity(data, word, max_n=None, focus=None):
    if max_n is not None:
        relations = get_entity_and_sort_rel(data, word, focus)[0:max_n]
        children = []  # list of child -> Node
        for rel in relations:
            children.append(Node(rel[1]).__dict__)
        return children
    entity = data[word]
    if entity is None:
        pass
        # print("can't find an entity for [{}] ".format(word))
    return entity


# 查找实体并排序关系
def get_entity_and_sort_rel(data, word, focus):
    def cmp(l1, l2):
        if l1[2] > l2[2]:
            return -1
        if l1[2] < l2[2]:
            return 1
        return 0

    raw = data[word]
    if raw is None:
        # print("can't find an entity for [{}] ".format(word))
        return []
    relations = []
    for key in raw.keys():
        if isinstance(raw[key], list) and ((focus is None) or (key in focus)):
            for rel in raw[key]:
                relations.append((key, rel[0], rel[1]))
    return sorted(relations, key=functools.cmp_to_key(cmp))





class Graph:
    def __init__(self, root, sentence, node, edge, adjacent, features):
        self.root = root  # 根节点
        self.sentence = sentence
        self.node = node  # 节点列表
        self.edge = edge
        self.adjacent_matrix = adjacent  # 邻接矩阵
        self.features = features  # 节点特征

    def get_adj(self):
        return self.adjacent_matrix

    def get_features(self):
        return self.features

    def get_root(self):
        return self.root

    def get_node(self):
        return self.node


def create_graph(data, entity, max_n, cur_hop, hop, focus):
    '''
    @param data: json file 包含所有实体关系
    @param entity: dict object
    @param max_n: 取权重最大的前几个边
    @param cur_hop: 0
    @param hop: 跳数
    @param focus: 表示关注的哪些关系 ['IsA', 'Synonym',...], 默认为全部
    @return:
    '''
    if cur_hop + 1 > hop:
        return
    # 获取子节点
    # print('当前在第 [{}] 跳, 正在构建 [{}] 的子图...'.format(cur_hop, entity['start']))
    entity['children'] = get_entity(data, entity['start'], max_n=max_n, focus=focus)
    for child in entity['children']:
        create_graph(data, child, max_n, cur_hop + 1, hop, focus)


def json_to_edges(json_obj, edges, nodes):
    start = json_obj['start']
    nodes.add(start)
    for end in json_obj['children']:
        edges.append((start, end['start']))
        json_to_edges(end, edges, nodes)
    return edges, nodes


def json_to_adj(json_obj, embeddings):
    '''
    @param json_obj: json object
    @param embeddings: 所有知识嵌入
    @return: Graph对象，包含所有需要的信息
    '''

    edge, nodes = [], set()
    edge, nodes = json_to_edges(json_obj, edge, nodes)
    nodes = list(nodes)
    adj = torch.zeros(len(nodes), len(nodes))
    for e in edge:
        x, y = nodes.index(e[0]), nodes.index(e[1])
        adj[x][y] = 1
        adj[y][x] = 1
    features = torch.Tensor(embeddings.get_sentence_word_list(nodes))
    return Graph(json_obj['start'], nodes, edge, adj, features)


def get_dependency_edges(document):
    '''
    获取句法依赖的依赖边
    @param document: spacy 的 Doc对象
    @return: 返回依赖边元组列表
    '''
    tokens = [token.text for token in document]
    seq_len = len(tokens)
    adj = torch.zeros(seq_len, seq_len)
    edges = []
    pos = []  # 词性标注
    for token in document:
        pos.append(token.pos_)
        for child in token.children:
            edges.append((token.text, child.text))
    return edges, pos


'''
det 限定词 90
noun 名词 92
aux 助动词 87
verb 动词 100
punct 标点符号 97
adp 介词 85
part 助词 94
adj 形容词 84
adv副词 86
'''


def creat_sentence_graph(sentence, spacy_nlp, data, embeddings, max_n, cur_hop=1, hop=2, focus=None, pos=None):
    '''
    @param pos: 指定那些詞性的詞需要擴充
    @param sentence: 句子，字符串
    @param spacy_nlp:
    @param data: json 文件 包含所有实体关系
    @param embeddings: 知识嵌入库
    @param max_n: 选取多少个关系
    @param cur_hop: 初始为0
    @param hop: 需要的跳数
    @param focus: 需要保留以的边关系列表如 ['IsA', 'Synonym', ...]， 默认为全部
    @return:
    '''
    document = spacy_nlp(sentence)
    dep_edges, dep_pos = get_dependency_edges(document)
    sentence = [token.text for token in document]
    edges = []
    all_node = []
    all_node.extend(sentence)
    edges.extend(dep_edges)
    idx = 0
    for word in sentence:
        if dep_pos[idx] not in pos:
            idx += 1
            continue
        idx += 1
        entity, edge, nodes = {'start': word, 'children': []}, [], set()
        create_graph(data, entity, max_n, cur_hop, hop, focus)  # 构建子图，返回一个嵌套字典 entity
        edge, nodes = json_to_edges(entity, edge, nodes)
        edges.extend(edge)
        all_node.extend(list(nodes))
    # node 去重
    tmp = []
    all_node = [tmp.append(i) for i in all_node if i not in tmp]
    all_node = tmp
    # print(all_node)
    adj = torch.zeros(len(all_node), len(all_node))
    for e in edges:
        x, y = all_node.index(e[0]), all_node.index(e[1])
        adj[x][y] = 1
        adj[y][x] = 1
    features = torch.Tensor(embeddings.get_sentence_word_list(all_node))
    return Graph(sentence[0], sentence, all_node, edges, adj, features)


class Entity:
    def __init__(self, start):
        self.start = start


def generate_json_file(data, file_path):
    json_dict = {}  # 存放str : Entity的映射, 查询复杂度为O(1)
    print("正在遍历 csv 文件...")
    t1 = time.time()
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        start = strip_end(row[2])
        rel = strip_relation(row[1])
        end = strip_end(row[3])
        weight = row[4]
        if start in json_dict:  # already exist
            if rel in json_dict[start]:
                json_dict[start][rel].append((end, weight))
            else:
                json_dict[start][rel] = []  # initialize a empty list
                json_dict[start][rel].append((end, weight))
        else:  # if dose not exist then create a Entity Object's dict
            entity = Entity(start).__dict__
            entity[rel] = []  # initialize a empty list
            entity[rel].append((end, weight))
            json_dict[start] = entity  # add the entity to dict
    t2 = time.time()
    print("耗时 {} 秒".format(t2 - t1))
    print("正在序列化字典...")
    t3 = time.time()
    json_str = json.dumps(json_dict, indent=4)
    t4 = time.time()
    print("耗时 {} 秒".format(t4 - t3))
    print("正在写入文件...")
    t5 = time.time()
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)
    t6 = time.time()
    print("耗时 {} 秒".format(t6 - t5))
    print("处理总耗时 {} 秒！".format(t6 - t1))


class DataDict(dict):
    def __missing__(self, key):
        return None


def visual_graph(graph):
    # 创建图
    tri_edge = get_matrix_triad(graph.get_adj())
    plt.figure(figsize=(10, 10))
    G = nx.Graph()
    cmap = plt.cm.get_cmap('plasma')
    G.add_nodes_from(range(len(graph.get_node())))
    G.add_weighted_edges_from(tri_edge)
    M = G.number_of_edges()
    S = len(graph.sentence)
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    # 绘图
    labels = {}
    i = 0
    for n in graph.get_node():
        labels[i] = n
        i = i + 1
    pos = nx.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_size=300, node_color="white", node_shape='s')
    # nx.draw(G, node_size=600, with_labels=True, pos=nx.circular_layout(G))
    edges = nx.draw_networkx_edges(G, pos=pos, node_size=300, edge_cmap=cmap, arrowstyle="->",
                                   arrowsize=10, width=2, edge_color=edge_colors)
    # for i in range(M):
    #     edges[i].set_alpha(edge_alphas[i])
    nx.draw_networkx(G, pos=pos, labels=labels, with_labels=True, node_color='white')
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def create_bert_graph(sentence, spacy_nlp, batch_len):
    doc = spacy_nlp(sentence)
    edges, pos = get_dependency_edges(doc)
    nodes = [token.text for token in doc]
    w_len = len(nodes)
    adj = torch.zeros(w_len, w_len)  # 构建邻接矩阵
    for edge in edges:
        x, y = nodes.index(edge[0]), nodes.index(edge[1])
        adj[x][y] = 1
        adj[y][x] = 1
    if adj.shape[0] > batch_len:  # 比bert输出长，截断
        adj = adj[0:batch_len][0:batch_len]
    elif adj.shape[0] < batch_len:  # 比bert输出短， 补齐
        adj = F.pad(adj, pad=(0, batch_len - adj.shape[0], 0, batch_len - adj.shape[0]), mode='constant', value=0)
    return adj


def get_knowledge_graph(input_ids, k_embeddings, nlp, tokenizer, entities_dict):
    graph_features = []
    graph_adjs = []
    # bert_graph_adjs = []
    for i in range(input_ids.shape[0]):
        _tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        sentence = tokenizer.convert_tokens_to_string(_tokens)
        # tokens = []
        # for w in _tokens:
        #     tokens.append(w.strip('#'))
        # bert_graph_adj = create_bert_graph(sentence, nlp, input_ids.shape[1])
        knowledge_graph = creat_sentence_graph(sentence=sentence, spacy_nlp=nlp, data=entities_dict,
                                               embeddings=k_embeddings, max_n=2, cur_hop=1, hop=2,
                                               focus=['IsA', 'Synonym', 'RelatedTo', 'DefinedAs', 'SimilarTo'],
                                               pos=['ADJ', 'NOUN', 'NUM', 'ADV'])
        graph_features.append(knowledge_graph.get_features())
        graph_adjs.append(knowledge_graph.get_adj())
        # bert_graph_adjs.append(bert_graph_adj)
    # 补全
    max_len = 0
    for t in graph_features:
        max_len = t.shape[0] if t.shape[0] > max_len else max_len
    for idx in range(len(graph_features)):
        if graph_features[idx].shape[0] < max_len:
            zero = torch.zeros((max_len - graph_features[idx].shape[0], graph_features[idx].shape[1]))
            graph_features[idx] = torch.cat([graph_features[idx], zero], dim=0)
            graph_adjs[idx] = F.pad(graph_adjs[idx],
                                    pad=(0, max_len - graph_adjs[idx].shape[0], 0, max_len - graph_adjs[idx].shape[0]),
                                    mode='constant', value=0)
    return graph_features, graph_adjs


'''
Spacy 命名体识别：
PERSON:      People, including fictional.
NORP:        Nationalities or religious or political groups.
FAC:         Buildings, airports, highways, bridges, etc.
ORG:         Companies, agencies, institutions, etc.
GPE:         Countries, cities, states.
LOC:         Non-GPE locations, mountain ranges, bodies of water.
PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
EVENT:       Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, etc.
LAW:         Named documents made into laws.
LANGUAGE:    Any named language.
DATE:        Absolute or relative dates or periods.
TIME:        Times smaller than a day.
PERCENT:     Percentage, including ”%“.
MONEY:       Monetary values, including unit.
QUANTITY:    Measurements, as of weight or distance.
ORDINAL:     “first”, “second”, etc.
CARDINAL:    Numerals that do not fall under another type.
'''
def expend_sentence(text, spacy_nlp, data, focus=None, pos=None):
    """
    @param focus:
    @param pos:
    @param text: 待扩展文本
    @param spacy_nlp: spacy 包
    @param data: 知识图谱
    @return: 扩展后文本
    """
    entity_map = {"TIME": "time", "DATE": "day", "CARDINAL": "number", "LOC": "location", "GPE": "location",
                  "FAC": "location"}
    keys = entity_map.keys()
    doc = spacy_nlp(text)
    ents = {}

    for ent in doc.ents:
        if ent.label_ in keys:
            ents[ent.end] = entity_map[ent.label_]
    displacy.render(doc, style='ent', jupyter=False)
    keys = ents.keys()
    tokens = [token for token in doc]
    exp_text = []
    # 遍历单词，择机扩展
    for idx in range(len(tokens)):
        exp_text.append(str(tokens[idx]))
        if idx+1 in keys:
            exp_text.append(str("(Is a " + ents[idx+1] + ")"))
    print(" ".join(exp_text))
    return " ".join(exp_text)








if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')
    embeddings = AttributeHeuristic('D:/Projects/Papers/mini.h5')
    with open("D:/Projects/Papers/entity_en.json", 'r', encoding='UTF-8') as f:
        t1 = time.time()
        entities_dict = json.load(f)
        entities_dict = DataDict(entities_dict)
        t2 = time.time()
        root = {'start': 'hotel', 'children': []}
        t3 = time.time()
        sentence = 'there is an expensive italian restaurant named frankie and bennys at cambridge leisure park clifton way cherry hinton . would you like to go there or choose another ?	great yeah that sounds great can you book a table for 5 people at 11:30 on sunday ?'
        expend_sentence(text=sentence, spacy_nlp=nlp, data=entities_dict, focus=['IsA'], pos=['NOUN'])

        # graph = creat_sentence_graph(sentence=sentence, spacy_nlp=nlp, data=entities_dict, embeddings=embeddings,
        #                              max_n=2, cur_hop=0, hop=1,
        #                              focus=['IsA', 'Synonym', 'RelatedTo', 'DefinedAs', 'SimilarTo'],
        #                              pos=['ADJ', 'NOUN', 'NUM', 'ADV'])
        # print(graph.get_features())
        # print(graph.get_features().shape)
        # print(graph.get_node())
        # torch.set_printoptions(profile="full")
        # print(graph.get_adj())
        # print(graph.get_root())
        # t4 = time.time()
        # print("构建邻接矩阵花费 {} 秒!".format(t4 - t3))
        # visual_graph(graph)
    # print("正在读取 CSV 文件...")
    # data = pd.read_csv('D:/Projects/Papers/assertions.csv', delimiter='\t')
    # data.columns = ['uri', 'relation', 'start', 'end', 'json']
    # data = data[data['start'].apply(lambda row: row.find('/en') > 0) & data['end'].apply(lambda row: row.find('/en') > 0)]
    # data.index = range(data.shape[0])
    # weights = data['json'].apply(lambda row: json.loads(row)['weight'])
    # data.pop('json')
    # data.insert(4, 'weights', weights)
    # print("总共有 [{}] 个实体!".format(data.shape[0]))
    # generate_json_file(data, './entity_en.json')

# root = Node('hotel')
# print("开始构图...")
# t1 = time.time()
# create_graph(root, 0, 1)
# t2 = time.time()
# print('给 [{}] 构图花费 {} 秒!'.format(root.word, (t2-t1)))
# print(root.__dict__)
# print(json.dumps(root))
#     with open('/data/lyh/kb/' + root['start'] + '.json', 'w') as json_file:
#         json_file.write(json.dumps(root, indent=4))
#     print("读取文件花费 {} 秒，构图花费 {} 秒！".format(t2 - t1, t3 - t2))
# with open("/data/lyh/kb/hotel.json", 'r', encoding='UTF-8') as f:
# print("读取文件花费 {} 秒，构图花费 {} 秒！".format(t2 - t1, t3 - t2))
# graph = json_to_adj(root, embeddings)
