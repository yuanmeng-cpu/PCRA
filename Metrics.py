# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:56:26 2019

@author: truthless
"""

#from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import sys
#sys.path.append("/storage/ysma/workspace/Dialog_sigir19/DCR/GCN/nltk/nltk/translate")
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
import torch.nn.functional as F
import math
import json

epsilon = 1e-10 # for numerical stability

def method4(p_n, references, hypothesis, hyp_len, *args, **kwargs):
    """
    Smoothing method 4:
    Shorter translations may have inflated precision values due to having
    smaller denominators; therefore, we give them proportionally
    smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
    suggests dividing by 1/ln(len(T)), where T is the length of the translation.
    """
    invcnt = 1
    for i, p_i in enumerate(p_n):
        if p_i.numerator == 0 and hyp_len != 0:
            incvnt = invcnt * 5 / np.log(hyp_len + epsilon)
            p_n[i] = 1 / incvnt
    return p_n

# original method4
#smoothie = SmoothingFunction().method4
smoothie = method4
#smoothie = "none"

EOS = 3


def distinct_metrics(hyps):
# outputs is a list which contains several sentences, each sentence contains several words
    unigram_count = 0
    bigram_count = 0
    trigram_count=0
    quagram_count=0
    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    quagram_set = set()
    for sen in hyps:
        for word in sen:
            unigram_count += 1
            unigram_set.add(word)
        for start in range(len(sen) - 1):
            bg = str(sen[start]) + ' ' + str(sen[start + 1])
            bigram_count += 1
            bigram_set.add(bg)
        for start in range(len(sen)-2):
            trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
            trigram_count+=1
            trigram_set.add(trg)
        for start in range(len(sen)-3):
            quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
            quagram_count+=1
            quagram_set.add(quag)
    dis1 = len(unigram_set) / len(hyps)#unigram_count
    dis2 = len(bigram_set) / len(hyps)#bigram_count
    dis3 = len(trigram_set)/len(hyps)#trigram_count
    dis4 = len(quagram_set)/len(hyps)#quagram_count
    return dis1, dis2, dis3, dis4


def bleu(refs, hyps):
    all_scores = []
    hyp_len1_cnt = 0
    hyp_len1_scores = []
    for ref, hyp in zip(refs, hyps):
        if EOS in ref:
            index = ref.index(EOS)
            ref = ref[:index]
        if EOS in hyp:
            index = hyp.index(EOS)
            hyp = hyp[:index]
        if smoothie == "none":
            score.append(sentence_bleu([ref], hyp))
        else:
            score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
            all_scores.append(score)
            if len(hyp) == 1:
                hyp_len1_cnt += 1
                hyp_len1_scores.append(score)

    return all_scores, hyp_len1_scores


def bleu_str(refs, hyps):
    all_scores = []
    hyp_len1_cnt = 0
    hyp_len1_scores = []
    for ref, hyp in zip(refs, hyps):
        if '<EOS>' in ref:
            index = ref.index('<EOS>')
            ref = ref[:index]
        if '<EOS>' in hyp:
            index = hyp.index('<EOS>')
            hyp = hyp[:index]
        if smoothie == "none":
            score.append(sentence_bleu([ref], hyp))
        else:
            score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
            all_scores.append(score)
            if len(hyp) == 1:
                hyp_len1_cnt += 1
                hyp_len1_scores.append(score)

    return all_scores, hyp_len1_scores


def bleu_rectified(refs, hyps):
    all_scores = []
    for ref, hyp in zip(refs, hyps):
        if EOS in ref:
            index = ref.index(EOS)
            ref = ref[:index]
        if EOS in hyp:
            index = hyp.index(EOS)
            hyp = hyp[:index]

        if len(hyp) == 1:
            score = 0
        else:
            score = sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4)
        all_scores.append(score)

    return all_scores


def bleu_str_rectified(refs, hyps):
    all_scores = []
    for ref, hyp in zip(refs, hyps):
        if '<EOS>' in ref:
            index = ref.index('<EOS>')
            ref = ref[:index]
        if '<EOS>' in hyp:
            index = hyp.index('<EOS>')
            hyp = hyp[:index]

        if len(hyp) == 1:
            score = 0
        else:
            score = sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4)
        all_scores.append(score)

    return all_scores


def bleu_str_corpus(refs, hyps):
    new_refs = []
    new_hyps = []
    for ref, hyp in zip(refs, hyps):
        if "<EOS>" in ref:
            index = ref.index("<EOS>")
            ref = ref[:index]
        if "<EOS>" in hyp:
            index = hyp.index("<EOS>")
            hyp = hyp[:index]
        new_refs.append([ref])
        new_hyps.append(hyp)

    if smoothie == "none":
        score = corpus_bleu(new_refs, new_hyps)
    else:
        score = corpus_bleu(new_refs, new_hyps, smoothing_function=smoothie)

    return score


def bleu_corpus(refs, hyps):
    new_refs = []
    new_hyps = []
    for ref, hyp in zip(refs, hyps):
        if EOS in ref:
            index = ref.index(EOS)
            ref = ref[:index]
        if EOS in hyp:
            index = hyp.index(EOS)
            hyp = hyp[:index]
        new_refs.append([ref])
        new_hyps.append(hyp)

    if smoothie == "none":
        score = corpus_bleu(new_refs, new_hyps)
    else:
        score = corpus_bleu(new_refs, new_hyps, smoothing_function=smoothie)

    return score


# used for multivoz evaluation
def bleu_str_sentence(val_file, hyps):
    assert len(val_file["sys"]) == len(hyps)
    all_scores = []
    for ref, hyp in zip(val_file["sys"], hyps):
        ref = ref.split(" ")
        if '_EOS' in ref:
            index = ref.index('_EOS')
            ref = ref[:index]
        hyp = hyp.split(" ")
        if '_EOS' in hyp:
            index = hyp.index('_EOS')
            hyp = hyp[:index]

        score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
        all_scores.append(score)

    return all_scores


def bleu_str_sentence_rectified(val_file, hyps):
    assert len(val_file["sys"]) == len(hyps)
    scores = []
    for ref, hyp in zip(val_file["sys"], hyps):
        ref = ref.split()
        if '_EOS' in ref:
            index = ref.index('_EOS')
            ref = ref[:index]
        hyp = hyp.split()
        if '_EOS' in hyp:
            index = hyp.index('_EOS')
            hyp = hyp[:index]
        if len(hyp) == 1:
            score = 0
        else:
            score = sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4)
        scores.append(score)

    return scores


def language_model_p(refs, word_p, lens):
    p = 0
    probs = F.softmax(word_p, dim=-1)
    for i, l in enumerate(lens):
        for j in range(l):
            p += np.log(probs[i, j, refs[i, j]].item() + 1e-100)
    return p


def perplexity(p, N):
    return np.exp(-1 / N * p)


def cal_acc(entity_p, entity_grd):
    entity_p = entity_p.cpu().numpy()
    entity_grd = entity_grd.cpu().numpy()

    ttl = 0
    correct = 0
    res = []
    for i, grds in enumerate(entity_grd):
        each_res = []
        for j, grd in enumerate(grds):
            grd = int(grd)
            pred = np.argmax(entity_p[i][j])
            each_res.append(pred)
            if grd == 0:
                continue
            else:
                ttl += 1
                if pred == grd:
                    correct += 1
        res.append(each_res)

    return ttl, correct, res


def cal_acc_new(entity_grd, outputs_grd, outputs_ori):
    entity_grd = entity_grd.cpu().numpy()
    entities = []
    for grds, ents in zip(outputs_grd, entity_grd):
        tmp_ents = set() 
        for word, ent in zip(grds, ents):
            if ent != 0:
                tmp_ents.add(word)
        entities.append(tmp_ents)

    ttl = 0
    correct = 0
    for ents, output_words in zip(entities, outputs_ori): 
        ttl += len(ents)
        for ent in ents:
            if ent in output_words:
                correct += 1

    return ttl, correct


def get_entity(entity_grd, outputs_grd,outputs_ori):
    '''
    for i,j in zip(outputs_grd,outputs_ori):
        print('i:{0},\n,j:{1}'.format(i,j))
        print('\n')
    '''
    '''
    for i,j in zip(outputs_grd,outputs_ori):
        print('i:{0},\n,j:{1}'.format(i,j))
        print('\n')
    '''
    entity_grd = entity_grd.cpu().numpy()
    entities = []
    for grds, ents in zip(outputs_grd, entity_grd):
        tmp_ents = []
        for word, ent in zip(grds, ents):
            if ent != 0:
                tmp_ents.append(word)
        entities.append(tmp_ents)
    ent_list = []
    for ents, output_words in zip(entities, outputs_ori):
        for ent in ents:
            if ent in output_words:
                ent_list.append(ent)
    return ent_list,entities


def NDCG(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 102))

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
        return dcg

    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
        dcg = dcg_at_n(current[i], n=k)
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg
        ndcgs.append(tmp_ndcg)
    return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return 1 / math.log(i+2)
    return 0


