import pandas as pd
import os
import time
import numpy as np
from deriveSummaryDUC import read_simMats, cluster_mat, oracle_per_cluster
import pickle
from collections import defaultdict
from utils import offset_str2list, offset_decreaseSentOffset, insert_string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-salience_pred_file', type=str)
parser.add_argument('-output_summ_dir', type=str)
parser.add_argument('-data_path', type=str)
parser.add_argument('-sim_mat_path', type=str)
parser.add_argument('-alignments_path', type=str)
args = parser.parse_args()



def find_abstractive_target(predictions_topic_cluster, alignments, topic):
    cluster_spans = list(predictions_topic_cluster['docSpanText'].values)
    alignments_cluster = alignments[(alignments['topic']==topic) & (alignments['docSpanText'].isin(cluster_spans))]
    aligned_summ_span_cands = list(alignments_cluster['summarySpanText'].drop_duplicates().values)
    summ_span_cands_score = []
    for summ_span in aligned_summ_span_cands:
        alignments_cluster_summ_span = alignments_cluster[alignments_cluster['summarySpanText'] == summ_span]
        summ_span_cands_score.append(alignments_cluster_summ_span['pred_prob'].sum())

    return aligned_summ_span_cands[np.argmax(summ_span_cands_score)]

def add_OIE_special_tok(docSpanOffsets, docSentCharIdx, sent):

    # document_tmp = document[:]
    span_offsets = offset_str2list(docSpanOffsets)
    offsets = offset_decreaseSentOffset(docSentCharIdx, span_offsets)
    # assume we have max 2 parts


    for offset in offsets[::-1]:  # [::-1] start from the end so the remain offsets won't be shifted
        sent = insert_string(sent, offset[1], ' > ')
        sent = insert_string(sent, offset[0], ' < ')


    return sent

    ##################################
######     main     ##############
##################################

if __name__ == "__main__":
    MAX_SENT = 100

    DATASETS = ['DUC2004']#['TAC2008','TAC2009','TAC2010']

    SET_TYPE = 'test'

    CLUSTERING = True
    SUMM_LEN = 100
    MAX_CLUSTERS = 10

    DUC2004_Benchmark = True


    FULL_SENT = False

    if FULL_SENT:
        full_sent_flag = '_full_sent'
    else:
        full_sent_flag = ''



    empty = 0
    analysis_list = []
    fusion_text = []
    fusion_target = []
    cluster_metadata = []


    metadata = pd.read_csv(
                './OIE_highlights/{}_{}_CDLM_fixed_truncated_metadata.csv'.format(
                    '_'.join(DATASETS),
                    SET_TYPE))

    predictions = pd.read_csv(args.salience_pred_file)

    assert (len(predictions) == len(metadata))
    metadata.insert(2, "prediction", predictions['prediction'])
    predictions = metadata

    for SET in DATASETS:

        if SET_TYPE != 'test':
            alignments = pd.read_csv(args.alignments_path)
            #'./dev{}_checkpoint-2000_negative.csv'.format(SET))

        sys_summary_path = args.output_summ_dir

        data_path = args.data_path + '/{}/'.format(SET)
        gold_summary_path = data_path + 'summaries/'







        for topic in os.listdir(data_path):
            print(topic)
            if topic == 'summaries':
                continue
            if SET.startswith('TAC'):
                topic = topic[:-3] + topic[-2:]



            summary = ''
            predictions_topic = predictions[predictions['topic'] == topic]

            if DUC2004_Benchmark:
                predictions_topic = predictions_topic[predictions_topic['prediction'] >= 0.4]
            else:
                predictions_topic = predictions_topic[predictions_topic['prediction'] >=  0.04]


            predictions_topic = predictions_topic.sort_values(by=['prediction'], ascending=False)

            if len(predictions_topic) == 0:
                empty += 1
                continue

            if CLUSTERING:
                simMat = read_simMats(topic, predictions_topic, SET, args.sim_mat_path)
                cluster_mat(simMat, predictions_topic['simMat_idx'].values, predictions_topic)

                oracle_per_cluster(SET, gold_summary_path, topic, predictions_topic, MAX_CLUSTERS)

                allowed_clusters = list(
                    predictions_topic.sort_values(by=['cluster_size', 'inFile_sentIdx'], ascending=[False, True])[
                        'cluster_idx'].drop_duplicates(keep="first").values)[:MAX_CLUSTERS]

                selected_spans = []

                summary = ' '

                for allowed_cluster_idx in allowed_clusters:
                    predictions_topic_cluster = predictions_topic[
                        predictions_topic['cluster_idx'] == allowed_cluster_idx]

                    predictions_topic_cluster = predictions_topic_cluster.sort_values(by=['prediction'],
                                                                                      ascending=False)
                    if len(predictions_topic_cluster) > 0:
                        if FULL_SENT:
                            predictions_topic_cluster['docSentText_special_tokens'] = predictions_topic_cluster.apply(lambda x: add_OIE_special_tok(x['docSpanOffsets'], x['docSentCharIdx'], x['docSentText']), axis=1)
                            fusion_text.append(
                                '<s> ' + ' </s> <s> '.join(
                                    list(predictions_topic_cluster['docSentText_special_tokens'].values)) + ' </s>')
                        else:
                            fusion_text.append(
                                '<s> ' + ' </s> <s> '.join(list(predictions_topic_cluster['docSpanText'].values)) + ' </s>')

                        if SET_TYPE != 'test':
                            fusion_target.append(find_abstractive_target(predictions_topic_cluster, alignments, topic))
                        else:
                            fusion_target.append('')


                        cluster_metadata.append([topic, list(predictions_topic_cluster.index)])
                        assert (predictions['docSpanText'].values[predictions_topic_cluster.index[0]]
                                == predictions_topic_cluster['docSpanText'].values[0])


    if DUC2004_Benchmark:
        out_dir = 'fusion_data/DUC2004{}/'.format(full_sent_flag)
    else:
        out_dir = 'fusion_data/TAC2011/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)



    cluster_metadata_df = pd.DataFrame(cluster_metadata, columns=['topic', 'cluster_indexes'])
    cluster_metadata_df.to_csv('{}/cluster_metadata_{}.csv'.format(out_dir,'_'.join(DATASETS)))

    if SET_TYPE == 'dev':
        SET_TYPE = 'val'


    with open('{}/{}.source'.format(out_dir, SET_TYPE), 'w') as f:
        f.write('\n'.join(fusion_text).replace('...', ' '))
    with open('{}/{}.target'.format(out_dir, SET_TYPE), 'w') as f:
        f.write('\n'.join(fusion_target).replace('...', ' '))

    if SET_TYPE =='test':
        # write "fake" files so the fusion model would run properly
        with open('{}/{}.source'.format(out_dir, 'val'), 'w') as f:
            f.write('\n'.join(fusion_text).replace('...', ' '))
        with open('{}/{}.target'.format(out_dir, 'val'), 'w') as f:
            f.write('\n'.join(fusion_text).replace('...', ' '))
        with open('{}/{}.target'.format(out_dir, 'test'), 'w') as f:
            f.write('\n'.join(fusion_text).replace('...', ' '))






