import pandas as pd
import os
from os.path import join
import sys
import time
from itertools import chain
sys.path.insert(1, '/home/nlp/ernstor1/rouge/SummEval_referenceSubsets/code_score_extraction')
import calculateRouge
import numpy as np
import glob
# from DataGenSalientIU_DUC_maxROUGE import greedy_selection_MDS, greedy_selection_clusters,  greedy_selection_all_clusters
import pickle
from sklearn.cluster import  AgglomerativeClustering
import re
from collections import defaultdict

def read_generic_file(filepath):
    """ reads any generic text file into
    list containing one line as element
    """
    text = []
    with open(filepath, 'r') as f:
        for line in f.read().splitlines():
            text.append(line.strip())
    return text

def write_summary(summary_path, summary, topic, type, ):
    SUMMARY_TYPES = {
        'gold': 'G',
        'system': 'S'}
    SUMMARY_LEN = 100

    type = SUMMARY_TYPES[type]
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)


    summary_name = str(topic) + '.M.' + str(SUMMARY_LEN) + '.T.' + type + '.html'
    with open(join(summary_path, summary_name), 'w') as outF:
        outF.write(summary)


def calc_rouge(gold_summary_path, sys_summary_path):
    calculateRouge.INPUTS = [(calculateRouge.COMPARE_SAME_LEN, gold_summary_path, sys_summary_path,
                              sys_summary_path + '0_rouge_scores.csv',
                              2002, calculateRouge.LEAVE_STOP_WORDS)]
    # calculateRouge.INPUTS = [(calculateRouge.COMPARE_VARYING_LEN, gold_summary_path, sys_summary_path,
    #            sys_summary_path + 'rouge_scores.csv',
    #            2002, calculateRouge.LEAVE_STOP_WORDS)]
    calculateRouge.main()


# the next *three* functions are taken from PreSumm implementation

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection_MDS(doc_sent_list, abstracts, summary_size=1000):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstracts = [_rouge_clean(abstract.lower().replace('...',' ... ')).split() for abstract in abstracts]
    # abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s.lower().replace('...',' ... ')).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]

    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]

    references_1grams = []
    references_2grams = []
    for abstract in abstracts:
        references_1grams.append(_get_word_ngrams(1, [abstract]))
        references_2grams.append(_get_word_ngrams(2, [abstract]))

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = np.mean([cal_rouge(candidates_1, reference_1grams)['f'] for reference_1grams in references_1grams])
            rouge_2 = np.mean([cal_rouge(candidates_2, reference_2grams)['f'] for reference_2grams in references_2grams])
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def greedy_selection_clusters(predictions_topic, abstracts, MAX_CLUSTERS = 9, summary_size=1000, HIGH_PRED_REPRESETATIVE_ORACLE = False, CLUSTERS_BY_ORDER = False):
    # HIGH_PRED_REPRESETATIVE_ORACLE = True
    # CLUSTERS_BY_ORDER = True

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    docSpanList = list(predictions_topic['docSpanText'].values)
    span_idx2cluster = list(predictions_topic['cluster_idx'].values)
    cluster_items = defaultdict(list)
    for span_idx, cluster_idx in enumerate(span_idx2cluster):
        cluster_items[cluster_idx].append(span_idx)

    allowed_clusters = list(predictions_topic.sort_values(by=['cluster_size', 'inFile_sentIdx'], ascending=[False, True])[
                                'cluster_idx'].drop_duplicates(keep="first").values)[:MAX_CLUSTERS]


    if HIGH_PRED_REPRESETATIVE_ORACLE:
        predictions_topic['original_idx2'] = range(len(predictions_topic))
        allowed_cluster_represetatives = []
        for allowed_cluster_idx in allowed_clusters:
            predictions_topic_cluster = predictions_topic[
                predictions_topic['cluster_idx'] == allowed_cluster_idx]

            predictions_topic_cluster = predictions_topic_cluster.sort_values(by=['prediction'], ascending=False)
            allowed_cluster_represetatives.append(predictions_topic_cluster.iloc[0]['original_idx2'])



    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstracts = [_rouge_clean(abstract.lower().replace('...',' ... ')).split() for abstract in abstracts]
    # abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s.lower().replace('...',' ... ')).split() for s in docSpanList]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]

    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]

    references_1grams = []
    references_2grams = []
    for abstract in abstracts:
        references_1grams.append(_get_word_ngrams(1, [abstract]))
        references_2grams.append(_get_word_ngrams(2, [abstract]))

    selected = []
    selected_clusters = []  #index of spans inside cluster that were already selected

    for s in range(summary_size):
        cur_max_rouge = 0#max_rouge#0
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            if (i in selected_clusters):
                continue

            if CLUSTERS_BY_ORDER:
                if (span_idx2cluster[i] != allowed_clusters[len(selected)]):
                    continue
            else:
                if (span_idx2cluster[i] not in allowed_clusters):
                    continue
            if HIGH_PRED_REPRESETATIVE_ORACLE:
                if i not in allowed_cluster_represetatives:
                    continue

            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = np.mean([cal_rouge(candidates_1, reference_1grams)['f'] for reference_1grams in references_1grams])
            rouge_2 = np.mean([cal_rouge(candidates_2, reference_2grams)['f'] for reference_2grams in references_2grams])
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)

        if len(selected) == MAX_CLUSTERS:
            return selected

        selected_clusters.extend(cluster_items[span_idx2cluster[cur_id]])
        max_rouge = cur_max_rouge

    return selected


def greedy_selection_all_clusters(predictions_topic, abstracts, MAX_CLUSTERS = 9, allowed_clusters = None, summary_size=1000, HIGH_PRED_REPRESETATIVE_ORACLE = True, CLUSTERS_BY_ORDER = False):
    # HIGH_PRED_REPRESETATIVE_ORACLE = False
    # CLUSTERS_BY_ORDER = True

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    docSpanList = list(predictions_topic['docSpanText'].values)
    span_idx2cluster = list(predictions_topic['cluster_idx'].values)
    cluster_items = defaultdict(list)
    for span_idx, cluster_idx in enumerate(span_idx2cluster):
        cluster_items[cluster_idx].append(span_idx)


    # allowed_clusters = list(predictions_topic[predictions_topic['cluster_size']>=3][
    #                             'cluster_idx'].drop_duplicates(keep="first").values)
    # if len(allowed_clusters) < 11:
    #     allowed_clusters = list(predictions_topic[predictions_topic['cluster_size'] >= 2][
    #                                 'cluster_idx'].drop_duplicates(keep="first").values)
    # # if len(allowed_clusters) < 10:
    # #     allowed_clusters = list(predictions_topic[predictions_topic['cluster_size'] >= 1][
    # #                                 'cluster_idx'].drop_duplicates(keep="first").values)

    # allowed_clusters = list(predictions_topic['cluster_idx'].drop_duplicates(keep="first").values)

    if allowed_clusters is None:
        #select all clusters
        allowed_clusters = list(predictions_topic['cluster_idx'].drop_duplicates(keep="first").values)

        # if len(predictions_topic[predictions_topic['cluster_size'] >= 3]['cluster_idx'].drop_duplicates()) < 10:
        #     allowed_clusters = list(predictions_topic[predictions_topic['cluster_size'] >= 2]['cluster_idx'].drop_duplicates().values)
        # else:
        #     allowed_clusters = list(predictions_topic[predictions_topic['cluster_size'] >= 3]['cluster_idx'].drop_duplicates().values)


    # if len(allowed_clusters)<MAX_CLUSTERS:
    #     allowed_clusters = list(
    #         predictions_topic.sort_values(by=['cluster_size', 'inFile_sentIdx'], ascending=[False, True])[
    #             'cluster_idx'].drop_duplicates(keep="first").values)[:MAX_CLUSTERS]


    if HIGH_PRED_REPRESETATIVE_ORACLE:
        predictions_topic['original_idx2'] = range(len(predictions_topic))
        allowed_cluster_represetatives = []
        for allowed_cluster_idx in allowed_clusters:
            predictions_topic_cluster = predictions_topic[
                predictions_topic['cluster_idx'] == allowed_cluster_idx]

            predictions_topic_cluster = predictions_topic_cluster.sort_values(by=['prediction'], ascending=False)
            allowed_cluster_represetatives.append(predictions_topic_cluster.iloc[0]['original_idx2'])



    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])
    abstracts = [_rouge_clean(abstract.lower().replace('...',' ... ')).split() for abstract in abstracts]
    # abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s.lower().replace('...',' ... ')).split() for s in docSpanList]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]

    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]

    references_1grams = []
    references_2grams = []
    for abstract in abstracts:
        references_1grams.append(_get_word_ngrams(1, [abstract]))
        references_2grams.append(_get_word_ngrams(2, [abstract]))

    selected = []
    selected_cluster_spans = []  #index of spans inside cluster that were already selected
    selected_rouge_diff = []
    selected_clusters = []
    for s in range(summary_size):
        cur_max_rouge = 0#max_rouge#0
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            if (i in selected_cluster_spans):
                continue

            if CLUSTERS_BY_ORDER:
                if (span_idx2cluster[i] != allowed_clusters[len(selected)]):
                    continue
            else:
                if (span_idx2cluster[i] not in allowed_clusters):
                    continue
            if HIGH_PRED_REPRESETATIVE_ORACLE:
                if i not in allowed_cluster_represetatives:
                    continue

            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = np.mean([cal_rouge(candidates_1, reference_1grams)['f'] for reference_1grams in references_1grams])
            rouge_2 = np.mean([cal_rouge(candidates_2, reference_2grams)['f'] for reference_2grams in references_2grams])
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):  #already selected all clusters
            assert(len(selected_rouge_diff)==len(allowed_clusters))
            return selected, selected_rouge_diff, selected_clusters
        selected.append(cur_id)
        selected_rouge_diff.append(cur_max_rouge-max_rouge)

        # if len(selected) == MAX_CLUSTERS:
        #     return selected

        selected_cluster_spans.extend(cluster_items[span_idx2cluster[cur_id]])
        selected_clusters.append(span_idx2cluster[cur_id])
        max_rouge = cur_max_rouge

    assert (len(selected_rouge_diff) == len(allowed_clusters))
    return selected, selected_rouge_diff, selected_clusters


def offset_str2list(offset):
    return [[int(start_end) for start_end in offset.split(',')] for offset in offset.split(';')]


def offset_decreaseSentOffset(sentOffset, scu_offsets):
    return [[start_end[0] - sentOffset, start_end[1] - sentOffset] for start_end in scu_offsets]


def Union(offsets, sentOffset):
    ranges_tmp = set([])
    for offset in offsets:
        offset = offset_str2list(offset)
        offset = offset_decreaseSentOffset(sentOffset, offset)
        ranges = [range(marking[0], marking[1]) for marking in offset]
        ranges = set(chain(*ranges))
        ranges_tmp = ranges_tmp | ranges
    return  ranges_tmp


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p.split('\n'):
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False

def read_simMats(topic_name, predictions_topic, dataset):
    sim_mats_path = './sim_mats/{}/'.format(dataset)


    if dataset.startswith('TAC'):
        topic_name = topic_name[:-2]
        topic_name = glob.glob(sim_mats_path+'/SupAligner_checkpoint-2000_'+ topic_name +'*-A' + '.pickle')[0][-15:-7]

    with open(os.path.join(sim_mats_path,'SupAligner_checkpoint-2000_'+ topic_name + '.pickle'), 'rb') as handle:
        simMat = pickle.load(handle)


    # simMat_l = np.tril(simMat) + np.tril(simMat).transpose()
    # np.fill_diagonal(simMat_l,0) #avoid summing diagonal twice
    # simMat_u = np.triu(simMat) + np.triu(simMat).transpose()
    # simMat = (simMat_l + simMat_u) / 2

    with open(os.path.join(sim_mats_path,topic_name + '_idx2span.pickle'), 'rb') as handle:
        idx2span = pickle.load(handle)

    span2idx = {}
    for key, value in idx2span.items():
        span2idx[value['documentFile'] + value['docScuText'] + str(value['docSentCharIdx'])] = key
    predictions_topic['simMat_idx'] = (predictions_topic['documentFile'] + predictions_topic['docSpanText']
                                               + predictions_topic['docSentCharIdx'].apply(str)).apply(lambda x: span2idx[x])




    return simMat











def createGT_labels(predictions_topic, data_path, topic, overSample=False):
        if overSample:
            labels_column_name = 'over_sample'
        else:
            labels_column_name = 'scnd_filter_label'

        predictions_topic['original_idx'] = range(len(predictions_topic))
        positive_alignments_topic = predictions_topic#[predictions_topic['pred_prob'] >= 0.5]

        abstracts = []
        # if DATASET == 'TAC2011':
        #     for summary_path in glob.iglob(data_path + topic.upper() + '.*'):
        #         summary = ' '.join(read_generic_file(summary_path))
        #         abstracts.append(summary)
        # else:
        for summary_path in glob.iglob(data_path + topic[:-1].upper() + '.*'):
                summary = ' '.join(read_generic_file(summary_path))
                abstracts.append(summary)

        docFile_summSpan_cands = list(positive_alignments_topic['docSpanText'].values)
        positive_summSpan_idx = greedy_selection_MDS(docFile_summSpan_cands, abstracts)
        positive_summSpan_original_idx = [positive_alignments_topic['original_idx'].values[cand_idx] for cand_idx in
                                          positive_summSpan_idx]
        scnd_filter_label = np.zeros(len(predictions_topic), dtype=int)
        scnd_filter_label[positive_summSpan_original_idx] = 1
        predictions_topic[labels_column_name] = scnd_filter_label

        ##validation for correct indexes
        docFile_summSpan_positive = [docFile_summSpan_cands[cand_idx] for cand_idx in positive_summSpan_idx]
        positive_labeled_spans_validation = predictions_topic[predictions_topic[labels_column_name] == 1][
            'docSpanText'].isin(docFile_summSpan_positive)
        assert (all(positive_labeled_spans_validation))

        return docFile_summSpan_positive



def cluster_mat(simMat, except_idx, predictions_topic):
    # zero_idx = np.delete(range(len(sim_mat)),except_idx)
    # sim_mat[zero_idx, :] = 0
    # sim_mat[:, zero_idx] = 0
    # except_idx = sorted(except_idx)
    sim_mat = simMat[except_idx, :]
    sim_mat = sim_mat[:, except_idx]
    sim_idx2new = {}
    for i in range(len(except_idx)):
        sim_idx2new[except_idx[i]] = i
    clustering = AgglomerativeClustering(affinity='precomputed',n_clusters=None, linkage="average" ,distance_threshold=0.5).fit(1-sim_mat)
    predictions_topic['cluster_idx'] = predictions_topic['simMat_idx'].apply(lambda x: clustering.labels_[sim_idx2new[x]])

    cluster_size = [list(clustering.labels_).count(i) for i in range(max(clustering.labels_)+1)]
    predictions_topic['cluster_size'] = predictions_topic['cluster_idx'].apply(lambda x: cluster_size[x])













def oracle_per_cluster(dataset, gold_summary_path, topic, predictions_topic, MAX_CLUSTERS, HIGH_PRED_REPRESETATIVE_ORACLE = False):
    abstracts = []
    if dataset.startswith('TAC'):
        for summary_path in glob.iglob(gold_summary_path + topic[:-2].upper() + '*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)
    else:
        for summary_path in glob.iglob(gold_summary_path + topic[:-1].upper() + '*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)

    assert(abstracts)
    docFile_summSpan_cands_idx = greedy_selection_clusters(predictions_topic, abstracts, MAX_CLUSTERS = MAX_CLUSTERS, HIGH_PRED_REPRESETATIVE_ORACLE  = HIGH_PRED_REPRESETATIVE_ORACLE)
    oracle_label = np.zeros(len(predictions_topic))
    oracle_label[docFile_summSpan_cands_idx] = 1
    predictions_topic['oracle_label'] = oracle_label

    return docFile_summSpan_cands_idx




def oracle_between_clusters(dataset, gold_summary_path, topic, predictions_topic, MAX_CLUSTERS):
    abstracts = []
    if dataset.startswith('TAC'):
        for summary_path in glob.iglob(gold_summary_path + topic[:-2].upper() + '*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)
    else:
        for summary_path in glob.iglob(gold_summary_path + topic[:-1].upper() + '*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)

    assert(abstracts)
    docFile_summSpan_cands_idx = greedy_selection_clusters(predictions_topic, abstracts, MAX_CLUSTERS = MAX_CLUSTERS, HIGH_PRED_REPRESETATIVE_ORACLE = True, CLUSTERS_BY_ORDER = False)
    oracle_label = np.zeros(len(predictions_topic))
    oracle_label[docFile_summSpan_cands_idx] = 1
    predictions_topic['oracle_label'] = oracle_label











def build_summary(prediction_topic_selected):
    summary = ''

    prediction_topic_selected_by_sent = prediction_topic_selected[['documentFile','docSentCharIdx']].drop_duplicates()
    for documentFile, docSentCharIdx in zip(prediction_topic_selected_by_sent['documentFile'].values,
                                            prediction_topic_selected_by_sent['docSentCharIdx'].values):
        selected_OIEs_sent = prediction_topic_selected[(prediction_topic_selected['documentFile'] == documentFile) &
                                               (prediction_topic_selected['docSentCharIdx'] == docSentCharIdx)]

        summary_indices = Union(selected_OIEs_sent['docSpanOffsets'].values, docSentCharIdx)
        summary_indices = sorted(list(summary_indices))
        sentenceText = selected_OIEs_sent['docSentText'].values[0]

        prev_idx = summary_indices[0]
        candidate_new_text = ''
        for idx in summary_indices:
            if idx == prev_idx + 1:
                candidate_new_text += sentenceText[idx]
            else:
                candidate_new_text += ' ' + sentenceText[idx]

            prev_idx = idx

        summary += candidate_new_text + '\n'  # add space between sentences

    return summary


def select_cluster_representative(prediction_topic_clusters_selected):
    selected_clusters = []

    prediction_topic_clusters_selected_concat = pd.concat(prediction_topic_clusters_selected,axis=0)
    prediction_topic_clusters_selected_concat = prediction_topic_clusters_selected_concat.sort_values(by=['prediction'], ascending=False)

    # prediction_topic_clusters_selected_concat = prediction_topic_clusters_selected_concat[prediction_topic_clusters_selected_concat['prediction'] > 0.1]

    selected_sents = pd.DataFrame(columns=prediction_topic_clusters_selected_concat.columns.to_list())

    for prediction_topic_cluster in prediction_topic_clusters_selected:
        cluster_idx = prediction_topic_cluster.iloc[0]['cluster_idx']
        if cluster_idx in selected_clusters:
            continue

        max_clusters_w_shared_sent = 0
        selected_sents_tmp = None
        for index, row in prediction_topic_cluster.iterrows():
            prediction_topic_clusters_selected_concat_sent = \
                prediction_topic_clusters_selected_concat[(prediction_topic_clusters_selected_concat['documentFile'] == row['documentFile']) &
                                                      (prediction_topic_clusters_selected_concat['docSentCharIdx'] == row['docSentCharIdx'])]
            prediction_topic_clusters_selected_concat_sent = prediction_topic_clusters_selected_concat_sent.drop_duplicates(['documentFile', 'docSentCharIdx', 'cluster_idx'])   #leave max one sentence per cluster (if there are two- leave the one with the highest prediction)
            if len(prediction_topic_clusters_selected_concat_sent) > max_clusters_w_shared_sent:
                max_clusters_w_shared_sent = len(prediction_topic_clusters_selected_concat_sent)
                selected_sents_tmp = prediction_topic_clusters_selected_concat_sent.copy()

        selected_sents = selected_sents.append(selected_sents_tmp)
        selected_clusters.extend(selected_sents_tmp['cluster_idx'].to_list())
        prediction_topic_clusters_selected_concat = prediction_topic_clusters_selected_concat[~prediction_topic_clusters_selected_concat['cluster_idx'].isin(selected_clusters)]    #remove selected clusters to avoid counting them again

    assert(len(selected_sents) == len(prediction_topic_clusters_selected))

    return selected_sents



def retrieve_R1_R2(sys_summary_path):
    full_path = os.path.join(sys_summary_path,'0_rouge_scores.csv')
    rouge_df = pd.read_csv(full_path)
    rouge_df = rouge_df.set_index('ROUGE_type')
    r1 = rouge_df['100_f']['R1']
    r2 = rouge_df['100_f']['R2']
    return r1, r2



    ##################################
######     main     ##############
##################################

if __name__ == "__main__":
  # tunning_list = []
  # for DUC_THRESH in np.linspace(0.0, 0.90, num=31):
  #  for CLUSTER_THRESH in  np.linspace(0.4, 0.7, num=7):
    MAX_SENT = 100

    DATASETS = ['DUC2004']#['TAC2008','TAC2009','TAC2010']
    SET_TYPE = 'test'
    ORACLE = False
    ORACLE_BY_CLUSTERS = False  #if True ORACLE_CLUSTER_REPRESENTATIVE or ORACLE_CLUSTER_RANKING must be True
    ORACLE_CLUSTER_REPRESENTATIVE = False   #take the best representative from each cluster (using original cluster ranking)
    ORACLE_CLUSTER_RANKING = False  #select best clusters
    ORACLE_BY_ALL_CLUSTERS = False  #select best clusters out of all clusters
    CLUSTERING = True

    SUMM_LEN = 100
    MAX_CLUSTERS = 10
    SENTENCE_LEVEL = False

    if ORACLE:
        oracle_flag = '_oracle'
    else:
        oracle_flag = ''



    sys_model = 'roberta'

    sys_checkpoint = 'checkpoint-1200'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
    sys_folder = 'OIE_TAC2008_TAC2009_2010_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_span_classifier_head_fixed'

    if SENTENCE_LEVEL:
        sys_checkpoint = 'checkpoint-1200'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
        sys_folder = 'OIE_full_TAC2008_TAC2009_2010_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_sentence_based_span_classifier_head'


    #DUC2004
    if DATASETS[0] == 'DUC2004':
        sys_checkpoint = 'checkpoint-1500'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
        sys_folder = 'OIE_DUC2003_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_span_classifier_head_fixed_finetuned_TAC8910'

        if SENTENCE_LEVEL:
        # #sentence-based
            sys_checkpoint = 'checkpoint-1800'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
            sys_folder = 'OIE_DUC2003_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_sentence_based_span_classifier_head_finetune_TAC8910_not_full/'#'OIE_DUC2003_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_sentence_based_span_classifier_head_finetune_TAC8910'






        ##full

    if DATASETS[0] == 'TAC2011':
        full_fixed = 'full'
    else:
        full_fixed = 'fixed'


    if DATASETS[0] =='DUC2003':
        sys_checkpoint = 'checkpoint-1500'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
        sys_folder = 'OIE_DUC2003_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_span_classifier_head_fixed_finetuned_TAC8910'

        metadata = pd.read_csv(
            './OIE_highlights/{}_{}_CDLM_greedyMaxRouge_no_alignment_{}_truncated_metadata.csv'.format(
            '_'.join(DATASETS),
            SET_TYPE, full_fixed))
    else:
        metadata = pd.read_csv(
            './OIE_highlights/{}_{}_CDLM_allAlignments_{}_truncated_metadata.csv'.format(
                '_'.join(DATASETS),
                SET_TYPE,full_fixed))


    predictions = pd.read_csv(
            './models/{}/{}/{}_{}_results_None.csv'.format(sys_folder,sys_checkpoint,
                SET_TYPE, '_'.join(DATASETS)))




    if SENTENCE_LEVEL:

        if DATASETS[0] == 'DUC2004':
            # # sentence_based duc
            metadata = pd.read_csv(
                './OIE_highlights/DUC2004_test_CDLM_greedyMaxRouge_no_alignment_sentence_based_fixed_truncated_metadata.csv')
        else:
            # #sentence_based
            metadata = pd.read_csv(
                './OIE_highlights/{}_{}_CDLM_allAlignments_sentence_based_full_truncated_metadata.csv'.format(
                    '_'.join(DATASETS),
                    SET_TYPE))




    assert (len(predictions)==len(metadata))
    metadata.insert(2, "prediction", predictions['prediction'])
    predictions = metadata

    len_pred = []
    len_pred_sent = []
    empty = 0
    analysis_list = []
    clusters_data = []

    prediction_selected = []

    for SET in DATASETS:

        sys_summary_path = './{}_system_summaries/{}/{}_'.format(SET,
                                                                                               sys_folder,
                                                                                               sys_checkpoint) + time.strftime(
            "%Y%m%d-%H%M%S") + '{}/'.format(oracle_flag)
        gold_summary_path = './data/{}/summaries/'.format(SET)

        data_path = './data/{}/'.format(SET)



        for topic in os.listdir(data_path):
            print(topic)
            if topic == 'summaries':
                continue
            if SET.startswith('TAC'):
                topic = topic[:-3] + topic[-2:]

            summary = ''
            predictions_topic = predictions[predictions['topic'] == topic]


            if SET =='DUC2004':
               predictions_topic = predictions_topic[predictions_topic['prediction'] >= 0.4]
            else:
                predictions_topic = predictions_topic[predictions_topic['prediction'] >= 0.04]
            #salience threshold -0.4 for DUC2004 0.04 for TAC2011



            predictions_topic = predictions_topic.sort_values(by=['prediction'], ascending=False)
            len_pred.append(len(predictions_topic))

            if len(predictions_topic) == 0:
                empty += 1
                continue

            if CLUSTERING:
                simMat = read_simMats(topic, predictions_topic, SET)
                cluster_mat(simMat, predictions_topic['simMat_idx'].values, predictions_topic)


                allowed_clusters = list(predictions_topic.sort_values(by=['cluster_size','inFile_sentIdx'], ascending=[False,True])[
                                            'cluster_idx'].drop_duplicates(keep="first").values)[:MAX_CLUSTERS]


                cluster_idx_idx = 0
                summary = ' '
                prediction_topic_selected = pd.DataFrame(columns=predictions_topic.columns.to_list())
                prediction_topic_clusters_selected = []
                # while len(summary.split(' ')) <= SUMM_LEN and cluster_idx_idx < len(allowed_clusters):
                while cluster_idx_idx < len(allowed_clusters):
                    predictions_topic_cluster = predictions_topic[predictions_topic['cluster_idx'] == allowed_clusters[cluster_idx_idx]]

                    # add most salient span from each cluster
                    predictions_topic_cluster = predictions_topic_cluster.sort_values(by=['prediction'], ascending=False)
                    new_cand = predictions_topic_cluster.iloc[0]
                    prediction_topic_selected = prediction_topic_selected.append(new_cand)

                    # select cluster representative span that its sentence appears in several clusters
                    # prediction_topic_clusters_selected.append(predictions_topic_cluster)
                    # #
                    # prediction_topic_selected = select_cluster_representative(prediction_topic_clusters_selected)



                    #if two selected spans are from the same sentence- take their "union"
                    summary = build_summary(prediction_topic_selected)

                    cluster_idx_idx += 1



            elif ORACLE:
                abstracts = []
                if SET.startswith('TAC'):
                    for summary_path in glob.iglob(gold_summary_path + topic[:-2].upper() + '*'):
                        abstract = ' '.join(read_generic_file(summary_path))
                        abstracts.append(abstract)
                else:
                    for summary_path in glob.iglob(gold_summary_path + topic[:-1].upper() + '.*'):
                        abstract = ' '.join(read_generic_file(summary_path))
                        abstracts.append(abstract)

                assert(abstracts)

                docFile_summSpan_cands = list(predictions_topic['docSpanText'].values)

                if ORACLE_BY_CLUSTERS:
                    read_simMats(topic, predictions_topic, SET)
                    cluster_mat(simMat, predictions_topic['simMat_idx'].values, predictions_topic)
                    if ORACLE_CLUSTER_REPRESENTATIVE:
                        docFile_summSpan_cands_idx = greedy_selection_clusters(predictions_topic, abstracts, MAX_CLUSTERS = MAX_CLUSTERS, CLUSTERS_BY_ORDER = True)

                    elif ORACLE_CLUSTER_RANKING:
                        docFile_summSpan_cands_idx = greedy_selection_clusters(predictions_topic, abstracts,
                                                                               MAX_CLUSTERS=MAX_CLUSTERS, HIGH_PRED_REPRESETATIVE_ORACLE = True)
                    else:
                        assert(False)
                    prediction_topic_selected = predictions_topic.iloc[docFile_summSpan_cands_idx]
                    summary = build_summary(prediction_topic_selected)
                elif ORACLE_BY_ALL_CLUSTERS:
                    read_simMats(topic, predictions_topic, SET)
                    cluster_mat(simMat, predictions_topic['simMat_idx'].values, predictions_topic)
                    docFile_summSpan_cands_idx, docFile_summSpan_cands_idx_rouge_diff,_ = greedy_selection_all_clusters(predictions_topic, abstracts, MAX_CLUSTERS = MAX_CLUSTERS)
                    prediction_topic_selected = predictions_topic.iloc[docFile_summSpan_cands_idx]
                    prediction_topic_selected['rouge_diff'] = docFile_summSpan_cands_idx_rouge_diff


                    summary = build_summary(prediction_topic_selected)

                    prediction_selected.append(prediction_topic_selected)


                else:
                    docFile_summSpan_cands_idx = greedy_selection_MDS(docFile_summSpan_cands, abstracts)
                    docFile_summSpan_positive = [docFile_summSpan_cands[cand_idx] for cand_idx in docFile_summSpan_cands_idx]
                    summary = ''
                    candidate_new_idx = 0
                    while len(summary.split(' ')) <= SUMM_LEN:
                        if candidate_new_idx >= len(docFile_summSpan_positive):
                            break
                        candidate_new_text = docFile_summSpan_positive[candidate_new_idx]

                        summary += candidate_new_text+"\n"
                        candidate_new_idx += 1

                oracle_label = np.zeros(len(docFile_summSpan_cands))
                oracle_label[docFile_summSpan_cands_idx] = 1
                predictions_topic['oracle_label'] = oracle_label


            else:

                predictions_topic = predictions_topic.sort_values(by=['prediction'], ascending=False)

                selected_spans = []

                candidate_new_idx = 0
                # analize_data(analysis_list)
                summary = ''
                while len(summary.split(' ')) <= SUMM_LEN and len(selected_spans) < MAX_SENT:


                    if candidate_new_idx >= len(predictions_topic):
                        break
                    candidate_new_text = predictions_topic['docSpanText'].values[candidate_new_idx]
                    candidate_new_idx += 1

                    if _block_tri(candidate_new_text, summary):
                        continue
                    selected_spans.append(candidate_new_text)
                    summary += candidate_new_text

                    # if '.' not in summary[-3:]:
                    #     summary += ' .\n'  # add period between sentences
                    # else:
                    #     summary += '\n'  # add space between sentences


                    summary += '\n'  # add space between sentences


            summary = summary.replace('...' ,' ')


            if SET.startswith('TAC'):
                write_summary(sys_summary_path, summary, topic=topic.upper()[:-2], type='system')
            else:
                write_summary(sys_summary_path, summary, topic=topic.upper()[:-1], type='system')

    calc_rouge(gold_summary_path, sys_summary_path)

    print('mean predictions per topic: ', np.mean(len_pred))
    print('max predictions per topic: ', max(len_pred))
    print('min predictions per topic: ', min(len_pred))
    print('num empty topic: ', empty)

  #   r1,r2 = retrieve_R1_R2(sys_summary_path)
  #   tunning_list.append([DUC_THRESH,CLUSTER_THRESH, r1,r2])
  #
  #
  #
  # tunning_df = pd.DataFrame(tunning_list, columns=['duc_thresh','cluster_thresh','R1', 'R2'])
  # tunning_df.to_csv('/home/nlp/ernstor1/highlighting/{}_system_summaries/{}/tunning/tunning_df.csv'.format(SET, sys_folder))




