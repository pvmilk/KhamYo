# -*- coding: utf-8 -*-
import itertools
import copy
import os
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pythainlp.tokenize import Tokenizer
from pythainlp.corpus import thai_words
from khamyo import __file__ as khamyo_file

path_file = os.path.join(os.path.dirname(khamyo_file),'data.json')

model = SentenceTransformer('kornwtp/ConGen-WangchanBERT-Small')

with open(path_file, encoding='utf-8') as fh:
    worddict = json.load(fh)

list_th = list(worddict.keys())
tokenizer = Tokenizer(list_th+list(thai_words()), engine='newmm')


def merge(l: list) -> list:
    list_sent = []
    temp = ""
    for i in l:
        if i not in list_th:
            temp += i
        elif temp!="":
            list_sent.append(temp)
            list_sent.append(i)
            temp = ""
        else:
            list_sent.append(i)
    if temp!="":
        list_sent.append(temp)
    return list_sent


def counts(l: list) -> defaultdict:
    _temp = defaultdict(int)
    for i in l:
        if i in list_th:
            _temp[i]+=1
    return _temp


def replace(sentence: str, top_k: int = 2) -> list:
    sent_words = tokenizer.word_tokenize(sentence)
    original_sent_words = copy.copy(sent_words)
    c = counts(sent_words)
    if c == {}:
        return [(sentence, None, [])]
    del c

    list_index = []
    list_temp = []
    
    single_subs = []

    j = 0
    for i,w in enumerate(sent_words):
        if w in list_th:
            if len(worddict[w])>1:
                list_index.append(i)
                list_temp.append(worddict[w])
            else:
                original_word = sent_words[i]
                new_word = worddict[w][0]
                sent_words[i] = new_word
                single_subs.append((original_word, new_word))

    sum_m = list(itertools.product(*list_temp))

    if not sum_m and single_subs:
        return [(''.join(sent_words), None, single_subs)]
    if not sum_m and not single_subs:
        return [(sentence, None, [])]

    list_sent_info = [] # Will store (sentence_str, substitutions)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    for i,v in enumerate(sum_m):
        _t = copy.copy(sent_words)
        
        current_subs = single_subs[:]
        for j,w in enumerate(v):
            _t[list_index[j]] = w
            original_word = original_sent_words[list_index[j]]
            current_subs.append((original_word, w))

        list_sent_info.append( (''.join(_t), current_subs) )

    list_sent = [info[0] for info in list_sent_info]

    if len(sum_m) <= 1:
        if not list_sent:
            return [(''.join(sent_words), None, single_subs)]
        s2 = model.encode(list_sent[0], convert_to_tensor=True)
        score = util.pytorch_cos_sim(sentence_embedding, s2)
        subs = list_sent_info[0][1] if list_sent_info else []
        return [(list_sent[0], score, subs)]

    corpus_embeddings = model.encode(list_sent, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
    
    # Check if cos_scores are on 'cuda'; if so, move them to 'cpu'.
    if cos_scores.is_cuda:
        cos_scores = cos_scores.to('cpu')

    top_results_indices = np.argpartition(-cos_scores, range(min(top_k, len(list_sent))))[0:top_k]
    
    final_results = []
    for i in top_results_indices.tolist():
        sent_str, subs = list_sent_info[i]
        score = cos_scores[i]
        final_results.append((sent_str, score, subs))
        
    return final_results
