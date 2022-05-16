import numpy as np
import re
import glob

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

def insert_OIE_special_tokens(text, offsets):
    # assume we have max 2 parts

    if len(offsets) > 1:  # start with the second offset, so first offset won't be changed by inserting.
        offsetB = offsets[1]
        if offsetB[1] > len(text):
            return None
        text = insert_string(text, offsetB[1], ' <OIE2_END> ')
        text = insert_string(text, offsetB[0], '<OIE2_START> ')

    offsetA = offsets[0]
    if offsetA[1] > len(text):
        return None
    text = insert_string(text, offsetA[1], ' <OIE1_END> ')
    text = insert_string(text, offsetA[0], ' <OIE1_START> ')

    return text

def special_tokens_idx(text, offsets, tokenizer):
    special_tokens_idx_list = []
    for offset_idx, offset in enumerate(offsets):
        for start_end_idx, start_end in enumerate(offset):
            special_token_prev_text = text[:start_end]
            special_token_idx = num_tokens(special_token_prev_text, tokenizer)
            special_token_idx += offset_idx * 2 + start_end_idx     #adding previous special tokens (that were not inserted yet)
            special_tokens_idx_list.append(special_token_idx)
    return special_tokens_idx_list

def words_to_token_ids(sent, tokenizer, stard_id=0):
    idx = stard_id
    enc = [tokenizer.encode(x, add_special_tokens=False) for x in sent.split()]
    desired_output = []
    for token in enc:
        tokenoutput = []
        for ids in token:
            tokenoutput.append(idx)
            idx += 1
        desired_output.append(tokenoutput)
    return desired_output


def num_tokens(text, tokenizer, add_special_tokens=False):
    # words_to_toks = words_to_token_ids(text, tokenizer)
    # all_tokens = [x for sublist in words_to_toks for x in sublist]
    all_tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return len(all_tokens)

def truncated_text_for_openie(text, tokenizer, MAX_SEQ_LEN = 512 , num_special_tokens = 5):
    words_to_toks = words_to_token_ids(text, tokenizer)
    all_tokens = [x for sublist in words_to_toks for x in sublist]

    if len(all_tokens) < MAX_SEQ_LEN - num_special_tokens:
        return text, tokenizer(text), len(all_tokens)
    last_word_idx = [i for i, sublist in enumerate(words_to_toks) for x in sublist if x == MAX_SEQ_LEN -1 - num_special_tokens][0]
    trunc_text = ' '.join(text.split()[:last_word_idx])
    return trunc_text, tokenizer(trunc_text)


def extract_global_attention_idx(text, tokenizer):
    special_tokens_idx_list = []

    #find special_token_char_idxes
    special_token_char_idxes = []
    special_token_char_idxes.append(text.find('<OIE1_START>') + len('<OIE1_START>'))
    special_token_char_idxes.append(text.find('<OIE1_END>')+ len('<OIE1_END>'))
    start_idx2 = text.find('<OIE2_START>')
    if start_idx2 > -1: #if exists
        special_token_char_idxes.append(start_idx2 + len('<OIE2_START>'))
        special_token_char_idxes.append(text.find('<OIE2_END>')+ len('<OIE2_END>'))

    # find special token idxes
    for special_token_char_idx in special_token_char_idxes:
        special_token_prev_text = text[:special_token_char_idx]
        special_token_idx = num_tokens(special_token_prev_text, tokenizer) # special token start sent included as we take len of tokens which is the idx+1
        assert(tokenizer.tokenize(text)[special_token_idx-1].startswith('<')) # check it finds the special token. special_token_idx-1 as we omit special start sent token, as tokemize function doesnt include it.
        assert(special_token_idx < 2048) #it shouldnt be longer then 2048 (0-2047), and the last token is special end of sentence token.
        special_tokens_idx_list.append(special_token_idx)

    return special_tokens_idx_list





# the next 4 functions are taken from PreSumm implementation

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



def ROUGE_selection(data_path, topic, predictions_topic):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    ROUGE_score = np.zeros(len(predictions_topic))


    doc_sent_list = list(predictions_topic['docSpanText'].values)
    pred_prob_list = list(predictions_topic['pred_prob'].values)

    abstracts = []
    for summary_path in glob.iglob(data_path + topic[:-1].upper() + '.*'):
        summary = ' '.join(read_generic_file(summary_path))
        abstracts.append(summary)

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

    aligned_rouge_scores = []
    for i, preb_prob in zip(range(len(sents)),pred_prob_list):
        if preb_prob < 0.5:
            ROUGE_score[i] = 0
        else:
            c = [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = np.mean(
                [cal_rouge(candidates_1, reference_1grams)['p'] for reference_1grams in references_1grams])
            rouge_2 = np.mean(
                [cal_rouge(candidates_2, reference_2grams)['p'] for reference_2grams in references_2grams])
            tmp_rouge_score = rouge_1 + rouge_2
            ROUGE_score[i] = tmp_rouge_score
            aligned_rouge_scores.append(tmp_rouge_score)

    ROUGE_median = np.percentile(aligned_rouge_scores,30)

    predictions_topic['ROUGE_score'] = ROUGE_score
    predictions_topic['scnd_filter_label'] = predictions_topic['ROUGE_score'].apply(lambda x: 1 if x >= ROUGE_median  else 0)



def read_abstracts(SET, SET_TYPE, topic):
    gold_summary_path = '/home/nlp/ernstor1/data/{}/{}/summaries/'.format(SET, SET_TYPE)

    abstracts = []
    if SET.startswith('TAC'):
        for summary_path in glob.iglob(gold_summary_path + topic[:-2].upper() + '.*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)
    else:
        for summary_path in glob.iglob(gold_summary_path + topic[:-1].upper() + '.*'):
            abstract = ' '.join(read_generic_file(summary_path))
            abstracts.append(abstract)

    assert (abstracts)
    return abstracts