import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
import os
import numpy as np
import re
import glob
from nltk import sent_tokenize
from utils import num_tokens
import math



def read_generic_file(filepath):
    """ reads any generic text file into
    list containing one line as element
    """
    text = []
    with open(filepath, 'r') as f:
        for line in f.read().splitlines():
            text.append(line.strip())
    return text

def offset_str2list(offset):
    return [[int(start_end) for start_end in offset.split(',')] for offset in offset.split(';')]

def offset_decreaseSentOffset(sentOffset, scu_offsets):
    return [[start_end[0] - sentOffset, start_end[1] - sentOffset] for start_end in scu_offsets]

def insert_string(string, index, value):
    return string[:index] + value + string[index:]




# the next *four* functions are taken from PreSumm implementation

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

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size=1000):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

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
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


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




def add_sent_special_tok(document, OIE_row = None):
    doc_sents = sent_tokenize(document)#[:20]
    if OIE_row is not None: #if main document
        doc_sents = doc_sents[:MAX_SENT_MAIN_DOC]
        sent_found_flag = False
        for sent_idx, sent in enumerate(doc_sents):
            if sent == OIE_row['docSentText']:
                sent_found_flag = True
                doc_sents[sent_idx] = add_OIE_special_tok(OIE_row['docSpanOffsets'], OIE_row['docSentCharIdx'], sent)
                if num_tokens('<doc-s> ' + '<s> ' + ' </s> <s> '.join(doc_sents[:sent_idx+1]) + ' </s>' + ' </doc-s>', tokenizer,
                              add_special_tokens=True)> MAX_TOKENS:
                    return None
                break


        if not sent_found_flag:
            return None
    else:   #if context document
        doc_sents = doc_sents[:MAX_SENT_CONTEXT_DOC]

    document = '<s> ' + ' </s> <s> '.join(doc_sents) + ' </s>'


    return document

def adding_files_context(file_context_combination, data_path, topic_dir):
    documents = []
    for file_context in file_context_combination:
        text = read_generic_file(os.path.join(data_path, topic_dir, file_context))
        document = " ".join(text)
        document = add_sent_special_tok(document)
        document = add_doc_special_tok(document)
        documents.append(document)
    context = ' '.join(documents)
    return context

def add_special_tok(row, document):
    document_tmp = document[:]#add_OIE_special_tok(docSpanOffsets, document)
    document_tmp = add_sent_special_tok(document_tmp, row)
    if document_tmp is not None:
        document_tmp = add_doc_special_tok(document_tmp)

    return document_tmp

def add_doc_special_tok(document_tmp):
    return '<doc-s> ' + document_tmp + ' </doc-s>'

def add_OIE_special_tok(docSpanOffsets, docSentCharIdx, sent, special_tokens_for_global_attn = True):

    # document_tmp = document[:]
    span_offsets = offset_str2list(docSpanOffsets)
    offsets = offset_decreaseSentOffset(docSentCharIdx, span_offsets)
    # assume we have max 2 parts

    if special_tokens_for_global_attn:
        for offset in offsets[::-1]: #[::-1] start from the end so the remain offsets won't be shifted
            sent = insert_string(sent, offset[1], ' <OIE1_END> ')
            sent = insert_string(sent, offset[0], ' <OIE1_START> ')
    else:
        for offset in offsets[::-1]: #[::-1] start from the end so the remain offsets won't be shifted
            sent = insert_string(sent, offset[1], ' > ')
            sent = insert_string(sent, offset[0], ' < ')



    return sent

def read_abstracts(DATASET, data_path, topic_dir):
    abstracts = []
    if DATASET.startswith('TAC'):
        # for summary_path in glob.iglob(
                # data_path + '/summaries/' + topic_dir[:-3].upper() + topic_dir[-2:].upper() + '.*'):
        for summary_path in glob.iglob(
                    data_path + '/summaries/' + topic_dir[:-3].upper() + '*'):
            summary = ' '.join(read_generic_file(summary_path))
            abstracts.append(summary)
    else:
        for summary_path in glob.iglob(data_path + '/summaries/' + topic_dir[:-1].upper() + '.*'):
            summary = ' '.join(read_generic_file(summary_path))
            abstracts.append(summary)
    return abstracts


def add_instance(full_instance, tokenizer, row, highlights_list, highlights_metadata_list, file_context_combination, alignment_label='alignment_label'):
    full_instance, global_attention_idx = extract_global_attention_idx(full_instance, tokenizer)


    print('num tokens:', num_tokens(full_instance, tokenizer, add_special_tokens=False))




    highlights_list.append([full_instance, row[alignment_label], global_attention_idx, row['greedyMaxRouge']])


    highlights_metadata_list.append(row.tolist()+ [file_context_combination])




def replace_special_token(text, special_token_char_idxes, old_special_token, new_special_token):
    text = text[:special_token_char_idxes[-1]] + new_special_token + text[special_token_char_idxes[-1] + len(
        old_special_token):]  # replace '<OIE1_START>' with '<'
    special_token_char_idxes[-1] += 1  # include new special token '<'
    return  text, special_token_char_idxes

def extract_global_attention_idx(text, tokenizer, model_max_tokens = None):

    if model_max_tokens is None:
        model_max_tokens = MAX_TOKENS

    #and replace new special tokens with '<' '>' so the model wont have to learn new tokens.
    special_tokens_idx_list = []
    special_token_char_idxes = []

    mark_start_idx = text.find('<OIE1_START>')
    while mark_start_idx > -1:
        # find special_token_char_idxes
        special_token_char_idxes.append(mark_start_idx)
        text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE1_START>', '<')
        special_token_char_idxes.append(text.find('<OIE1_END>'))
        text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE1_END>', '>')
        mark_start_idx = text.find('<OIE1_START>')

    # #find special_token_char_idxes
    # special_token_char_idxes = []
    # special_token_char_idxes.append(text.find('<OIE1_START>'))
    # text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE1_START>', '<')
    # special_token_char_idxes.append(text.find('<OIE1_END>'))
    # text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE1_END>', '>')
    # start_idx2 = text.find('<OIE2_START>')
    # if start_idx2 > -1: #if exists
    #     special_token_char_idxes.append(start_idx2)
    #     text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE2_START>', '<')
    #     special_token_char_idxes.append(text.find('<OIE2_END>'))
    #     text, special_token_char_idxes = replace_special_token(text, special_token_char_idxes, '<OIE2_END>', '>')

    # find special token idxes
    for special_token_char_idx in special_token_char_idxes:
        special_token_prev_text = text[:special_token_char_idx]
        special_token_idx = num_tokens(special_token_prev_text, tokenizer) # special token start sent included as we take len of tokens which is the idx+1
        assert(('<' in tokenizer.tokenize(text)[special_token_idx-1]) or ('>' in tokenizer.tokenize(text)[special_token_idx-1])) # check it finds the special token. special_token_idx-1 as we omit special start sent token, as tokemize function doesnt include it.
        assert(special_token_idx < model_max_tokens) #it shouldnt be longer then 2048 (0-2047), and the last token is special end of sentence token.
        special_tokens_idx_list.append(special_token_idx)

    return text, special_tokens_idx_list


def createGT_labels(OIEs_topic, data_path, topic_dir, DATASET):
    labels_column_name = 'greedyMaxRouge'


    OIEs_topic['original_idx'] = range(len(OIEs_topic))



    abstracts = read_abstracts(DATASET, data_path, topic_dir)

    docFile_summSpan_cands = list(OIEs_topic['docSpanText'].values)
    positive_summSpan_idx = greedy_selection_MDS(docFile_summSpan_cands, abstracts)
    positive_summSpan_original_idx = [OIEs_topic['original_idx'].values[cand_idx] for cand_idx in positive_summSpan_idx]
    scnd_filter_label =  np.zeros(len(OIEs_topic), dtype=int)
    scnd_filter_label[positive_summSpan_original_idx] = 1

    if labels_column_name in OIEs_topic.columns:

        scnd_filter_label = np.array(OIEs_topic[labels_column_name].to_list()) + scnd_filter_label


    OIEs_topic[labels_column_name] = scnd_filter_label


    ##validation for correct indexes
    positive_labeled_spans = OIEs_topic[OIEs_topic[labels_column_name] == 1]['docSpanText'].to_list()
    positive_labeled_spans_validation = [docFile_summSpan_cands[cand_idx] in positive_labeled_spans for cand_idx in positive_summSpan_idx]


    assert(all(positive_labeled_spans_validation))


    return OIEs_topic

def add_sent_in_file_idx(OIEs_topic, data_path, topic_dir):
    doc_sent_idx = np.zeros(len(OIEs_topic), dtype=int)

    OIEs_topic['original_idx'] = range(len(OIEs_topic))

    topic_files = os.listdir(os.path.join(data_path, topic_dir))
    for file_idx, file in enumerate(topic_files):
        OIEs_topic_file = OIEs_topic[OIEs_topic['documentFile']==file]
        text = read_generic_file(os.path.join(data_path, topic_dir, file))
        document = " ".join(text)
        doc_sents = sent_tokenize(document)
        for sent_idx, doc_sent in enumerate(doc_sents):
            OIEs_topic_file_sent_original_idx = (OIEs_topic_file['original_idx'][OIEs_topic_file['docSentText'] == doc_sent]).values
            doc_sent_idx[OIEs_topic_file_sent_original_idx] = sent_idx


    OIEs_topic['inFile_sentIdx'] = doc_sent_idx
    return OIEs_topic


def positive_augmentation(num_negative, num_positive, highlights_df, highlights_metadata_df, label_tag = 'label', SAFE_BUFFER = 100):
    original_len_highlights_df = len(highlights_df)
    augmentation_factor = (num_negative- num_positive - SAFE_BUFFER)/num_positive
    if label_tag != 'label':
        augmentation_factor = (num_negative - num_positive - SAFE_BUFFER) / len(highlights_df[highlights_df[label_tag]==1])
    #threshold = 0.75
    augmentation_factor = math.floor(augmentation_factor) #if augmentation_factor < (math.floor(augmentation_factor) + threshold) else math.ceil(augmentation_factor)
    positive_highlights_df = highlights_df[highlights_df[label_tag] == 1]
    positive_highlights_metadata_df = highlights_metadata_df.loc[positive_highlights_df.index, :]
    if augmentation_factor >= 1:



        for i in range(augmentation_factor):
            highlights_df = highlights_df.append(positive_highlights_df)
            highlights_metadata_df = highlights_metadata_df.append(positive_highlights_metadata_df)



    num_negative = len(highlights_df[highlights_df['label'] == 0])
    num_positive = len(highlights_df[highlights_df['label'] == 1])

    print('negative samples:', num_negative)
    print('positive samples:', num_positive)

    # augmentation_factor = (num_negative - num_positive) / num_positive      # if still not equal- add part of positive samples.
    # if augmentation_factor > 0.5:
    if num_negative - num_positive > SAFE_BUFFER:
        selected_index = np.random.choice(positive_highlights_df.index.to_list(),num_negative - num_positive -SAFE_BUFFER,replace=False)
        selected_positive_highlights_df = highlights_df[:original_len_highlights_df].loc[selected_index, :] #copy from original highlights_df (before augmentation) so rows won't be double augmented by their index
        selected_positive_highlights_metadata_df = highlights_metadata_df[:original_len_highlights_df].loc[selected_index, :]
        highlights_df = highlights_df.append(selected_positive_highlights_df)
        highlights_metadata_df = highlights_metadata_df.append(selected_positive_highlights_metadata_df)

    num_negative = len(highlights_df[highlights_df['label'] == 0])
    num_positive = len(highlights_df[highlights_df['label'] == 1])

    print('negative samples:', num_negative)
    print('positive samples:', num_positive)

    return highlights_df, highlights_metadata_df






##################################
######     main     ##############
##################################
if __name__ == "__main__":
    np.random.seed(42)
    SET = 'train'
    DATASETS =  ['TAC2008','TAC2009','TAC2010']
    NUM_CONTEXT_FILES = 9
    MAX_TOKENS = 4096
    filter_negative = False
    FILTER_RATE = 0.4
    over_sample_positive = False
    MAX_SENT_MAIN_DOC = 20
    MAX_SENT_CONTEXT_DOC = 9
    sentences_level = False



    if SET == 'train':
        filter_negative = True
        over_sample_positive = True



    positive_label = 'greedyMaxRouge'



    if filter_negative:
        filter_negative_label = '_filter_negative'
    else:
        filter_negative_label = ''



    if over_sample_positive:
        over_sample_positive_label = '_over_sample_positive'
    else:
        over_sample_positive_label = ''

    if sentences_level:
        sentences_level_label = '_sentence_based'
    else:
        sentences_level_label = ''





    OUTPUT_PATH = 'OIE_highlights/{}_{}_CDLM{}{}{}_fixed_truncated.csv'.format("_".join(DATASETS), SET,
                                                                                             filter_negative_label,
                                                                                             over_sample_positive_label,
                                                                                             sentences_level_label)
    highlights_list = []
    highlights_metadata_list = []

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('./CDLM/')

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<OIE1_START>', '<OIE1_END>', '<OIE2_START>', '<OIE2_END>']})

    for DATASET in DATASETS:
        data_path = 'data/{}/'.format(DATASET)
        OIEs = pd.read_csv('OIE_cands/OIE_cands_{}.csv'.format(DATASET))

        if sentences_level:
            OIEs['docSpanText'] = OIEs['docSentText']
            OIEs['docSpanOffsets'] = OIEs['docSentCharIdx'].apply(str) + ', ' + (
                        OIEs['docSentCharIdx'] + OIEs['docSentText'].apply(len)).apply(str)



        used_positive_spans = 0

        for topic_dir in os.listdir(data_path):
            print(topic_dir)
            if topic_dir == 'summaries':
                continue

            OIEs_topic = OIEs[OIEs['topic'] == topic_dir]

            if DATASET.startswith('TAC'):
                topic_dir_tac2011 = topic_dir[:-3].upper() + topic_dir[-2:].upper()
                OIEs_topic = OIEs[OIEs['topic'] == topic_dir_tac2011]


            OIEs_topic = add_sent_in_file_idx(OIEs_topic, data_path, topic_dir)
            OIEs_topic = OIEs_topic[OIEs_topic['inFile_sentIdx'] < MAX_SENT_MAIN_DOC]
            OIEs_topic = createGT_labels(OIEs_topic, data_path, topic_dir, DATASET)



            topic_files = os.listdir(os.path.join(data_path, topic_dir))
            topic_dates = [topic[re.search(r"\d", topic).start():] for topic in topic_files]#[topic_file[3:] for topic_file in topic_files]
            topic_files = [x for _, x in sorted(zip(topic_dates, topic_files))]
            for file_idx, file in enumerate(topic_files):

                text = read_generic_file(os.path.join(data_path, topic_dir, file))
                document = " ".join(text)

                post_context_files = topic_files[file_idx + 1:file_idx + 1 + NUM_CONTEXT_FILES]
                pre_context_files = []
                if len(post_context_files) < NUM_CONTEXT_FILES:
                    diff_len = NUM_CONTEXT_FILES - len(post_context_files)
                    pre_context_files = topic_files[max(0, file_idx - diff_len):file_idx]  # + context_files

                assert (len(post_context_files + pre_context_files) == min(NUM_CONTEXT_FILES, len(topic_files) - 1))

                # trunced_document = truncated_text_for_openie(document, tokenizer)

                OIEs_topic_docFile = OIEs_topic[
                    OIEs_topic['documentFile'] == file]

                for index, row in OIEs_topic_docFile.iterrows():

                    main_document = add_special_tok(row, document)
                    if main_document is None:
                        continue
                    if row[positive_label]:
                        used_positive_spans += 1

                    else:
                        if filter_negative:
                            if np.random.choice([0, 1], p=[FILTER_RATE,
                                                           1 - FILTER_RATE]):  # 'continue' in random (1 - FILTER_RATE) of negative cases.
                                continue

                    # for file_context_combination in [context_files]:# combinations(topic_files_tmp,NUM_CONTEXT_FILES):  # all context combinations of 2 files

                    pre_documents_context = adding_files_context(pre_context_files, data_path, topic_dir)
                    post_documents_context = adding_files_context(post_context_files, data_path, topic_dir)
                    file_context_combination = pre_context_files + post_context_files

                    full_instance = pre_documents_context + ' ' + main_document + ' ' + post_documents_context


                    add_instance(full_instance, tokenizer, row, highlights_list,
                                 highlights_metadata_list, file_context_combination, alignment_label=positive_label)



    print(len(highlights_list))

    highlights_df = pd.DataFrame(highlights_list, columns=['', 'label', 'global_attention_idx', 'greedyMaxRouge'])
    highlights_metadata_df = pd.DataFrame(highlights_metadata_list,
                                          columns=OIEs_topic.columns.tolist() + ['doc_context'])

    num_negative = len(highlights_df[highlights_df['label'] == 0])
    num_positive = len(highlights_df[highlights_df['label'] == 1])

    print('negative samples:', num_negative)
    print('positive samples:', num_positive)

    if over_sample_positive:
        highlights_df, highlights_metadata_df = positive_augmentation(num_negative, num_positive, highlights_df,
                                                                      highlights_metadata_df)

    highlights_df = highlights_df[['', 'label', 'global_attention_idx']]

    highlights_df.to_csv(OUTPUT_PATH, index=False)
    highlights_metadata_df.to_csv(OUTPUT_PATH[:-4] + '_metadata.csv', index=False)
