import pandas as pd
import time
from deriveSummaryDUC import read_generic_file, write_summary, calc_rouge, build_summary, greedy_selection_MDS
import numpy as np
from utils import read_abstracts


def most_similar_text(ref_text, cands):
    ref_text_toks = ref_text.split()
    sim_score = []
    for cand in cands:
        sum = 0
        cand_toks = cand.split()
        for tok in cand_toks:
            if tok in ref_text_toks:
                sum += 1
        sim_score.append(sum)

    return np.argmax(sim_score)

def replace_with_extractive(cluster_indexes, predictions_topic, pred_sent):
    cluster_indexes = [int(idx) for idx in cluster_indexes[1:-1].split(',')]
    predictions_topic_cluster = predictions_topic[predictions_topic.index.isin(cluster_indexes)]
    predictions_topic_cluster = predictions_topic_cluster.sort_values(by=['prediction'], ascending=False)
    doc_cluster_tokens = set((' '.join(predictions_topic_cluster['docSpanText'].values.tolist())).lower().split())
    pred_sent_tokens = set(pred_sent.lower().split())
    if len(doc_cluster_tokens.intersection(pred_sent_tokens)) < 0.3*len(pred_sent_tokens):
        return predictions_topic_cluster['docSpanText'].values.tolist()[0]
    else:

        return pred_sent





    ##################################
######     main     ##############
##################################

if __name__ == "__main__":

    SET = 'TAC2011'
    SET_TYPE = 'test'

    EXTRACTIVE = False
    DUC2004_Benchmark = False

    ORACLE = False



    model_name = 'greedyMaxRouge'
    sys_checkpoint = 'checkpoint-1200'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
    sys_folder = 'OIE_TAC2008_TAC2009_2010_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_span_classifier_head_fixed'

    ##DUC2004
    if DUC2004_Benchmark:
        sys_checkpoint = 'checkpoint-1500'  # 'checkpoint-180'#'checkpoint-540'#'checkpoint-1020'#'checkpoint-540'#'checkpoint-600'  #'checkpoint-1140'#'checkpoint-240'#'checkpoint-180'  # 'checkpoint-1080'
        sys_folder = 'OIE_DUC2003_highlighter_CDLM_greedyMaxRouge_no_alignment_filter_negative_over_sample_positive_span_classifier_head_fixed_finetuned_TAC8910'

    ## reading files and initializations

    sys_summary_path = './{}_system_summaries/{}/{}_'.format(SET, sys_folder,
                                                                                           sys_checkpoint) + time.strftime(
        "%Y%m%d-%H%M%S" + '/')
    gold_summary_path = './data/{}/summaries/'.format(SET)



    fusion_data_dir = 'fusion_data/{}/{}/'.format(SET, model_name)
    cluster_metadata = pd.read_csv('{}/cluster_metadata_{}.csv'.format(fusion_data_dir,SET))
    cluster_metadata = cluster_metadata.drop(len(cluster_metadata)-1)

    fusion_out_dir = './fusion_output/{}/'.format(SET)


    fusion_predictions_path = fusion_out_dir + 'test_generations.txt'

    fusion_predictions = read_generic_file(fusion_predictions_path)

    cluster_metadata['fusion_prediction'] = fusion_predictions


    if DUC2004_Benchmark:
        metadata = pd.read_csv(
            './OIE_highlights/{}_{}_CDLM_allAlignments_fixed_truncated_metadata.csv'.format(
                SET,
                SET_TYPE))
    else:
        metadata = pd.read_csv(
        './OIE_highlights/{}_{}_CDLM_allAlignments_full_truncated_metadata.csv'.format(
            SET,
            SET_TYPE))
    predictions = pd.read_csv(
        './models/out_final/{}/{}/{}_{}_results_None.csv'.format(sys_folder, sys_checkpoint,
                                                                                   SET_TYPE, SET))

    assert (len(predictions) == len(metadata))
    metadata.insert(2, "prediction", predictions['prediction'])
    predictions = metadata





    ##start summary building

    for topic in sorted(set(cluster_metadata['topic'].values)):
        print(topic)
        cluster_metadata_topic = cluster_metadata[cluster_metadata['topic'] == topic]

        if EXTRACTIVE:
            predictions_topic = predictions[predictions['topic'] == topic]
            prediction_topic_selected = pd.DataFrame(columns=predictions_topic.columns.to_list())
            summary = ''
            for cluster_indexes, fusion_prediction in zip(cluster_metadata_topic['cluster_indexes'].values, cluster_metadata_topic['fusion_prediction'].values):
                cluster_indexes = [int(idx) for idx in cluster_indexes[1:-1].split(',')]
                predictions_topic_cluster = predictions_topic[predictions_topic.index.isin(cluster_indexes)]
                candidate_new_text_idx = most_similar_text(fusion_prediction, list(predictions_topic_cluster['docSpanText'].values))

                prediction_topic_selected = prediction_topic_selected.append(predictions_topic_cluster.iloc[candidate_new_text_idx])
                # summary += candidate_new_text + '\n'
                summary = build_summary(prediction_topic_selected)


        else:



            #if limited for 100 words
            summary = ''
            selected_sents = []
            num_words = 0


            if ORACLE:
                abstracts = read_abstracts(SET, SET_TYPE, topic)
                selected_idx = greedy_selection_MDS(cluster_metadata_topic['fusion_prediction'].values, abstracts)
                cluster_metadata_topic_selected = cluster_metadata_topic.iloc[selected_idx]
                summary = '\n'.join(cluster_metadata_topic_selected['fusion_prediction'].values)
            else:



                for cluster_indexes, pred_sent in zip(cluster_metadata_topic['cluster_indexes'].values,
                                                                  cluster_metadata_topic['fusion_prediction'].values):


                    summary += pred_sent + '\n'

                    num_words += len(pred_sent.split(' '))
                    if num_words > 100:
                        break


                    selected_sents.append(pred_sent)


        summary = summary.replace('...', ' ')



        if SET.startswith('TAC'):
            write_summary(sys_summary_path, summary, topic=topic.upper()[:-2], type='system')
        else:
            write_summary(sys_summary_path, summary, topic=topic.upper()[:-1], type='system')

    calc_rouge(gold_summary_path, sys_summary_path)