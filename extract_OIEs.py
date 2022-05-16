import os
from utils import read_generic_file
from nltk import tokenize
from supervised_oie_wrapper.run_oie import run_oie
import copy
import pandas as pd



def offset_list2str(list):
    return ';'.join(', '.join(map(str, offset)) for offset in list)

def checkPairContained(containedCandidateOffset_list, containOffset_list):
    containedList = []
    for containedCandidate in containedCandidateOffset_list:
        contained = False
        for offset in containOffset_list:
            contained_start, contained_end = containedCandidate
            start, end = offset
            if contained_start >= start and contained_end <= end:
                contained = True
        containedList.append(contained)

    notContained = not(all(containedList))  #if all spans are contained
    return notContained


def checkContained(scuOffsetDict,sentenceText, sentenceOffset = 0):
    notContainedDict = {}
    for containedCandidate, containedCandidateOffset_list in scuOffsetDict.items():
        notContainedList = []
        for contain, containOffset_list in scuOffsetDict.items():
            if contain == containedCandidate:
                continue

                #if one of scus is the full sentence, don't filter the other scus.
            full_sent_scu = True if containOffset_list[0][0] - sentenceOffset == 0 and\
                    containOffset_list[0][1] - sentenceOffset > 0.95*(len(sentenceText) - 1) else False
            if full_sent_scu:
                continue
            notContained = checkPairContained(containedCandidateOffset_list, containOffset_list)
            notContainedList.append(notContained)
            # if not notContained:
            #     print(containedCandidate)
            #     print (contain)

        notContainedDict[containedCandidate] = all(notContainedList)

    return notContainedDict


def generate_scu_oie_multiSent(sentences, doc_summ='summ'):
    """ Given a scu sentence retrieve SCUs"""

    if doc_summ=='summ':
        KEY_sent = 'scuSentence'
        KEY_sent_char_idx = 'scuSentCharIdx'
        KEY_scu_text = 'scuText'
        KEY_scu_offset = 'scuOffsets'
    else:
        KEY_sent = 'docSentText'
        KEY_sent_char_idx = 'docSentCharIdx'
        KEY_scu_text = 'docScuText'
        KEY_scu_offset = 'docScuOffsets'

    _, oies = run_oie([sentence[KEY_sent] for sentence in sentences], cuda_device = 0)
    #adaptation for srl
    # oies = []
    # for sentence in sentences:
    #     oies.append(predictor.predict(sentence = sentence[KEY_sent] ))


    scu_list = []
    assert(len(sentences) == len(oies))
    for sentence ,oie in zip(sentences,oies):
        sentence[KEY_sent] = sentence[KEY_sent].replace(u'\u00a0', ' ')
        # ipdb.set_trace()
        if not oie:  # if list is empty
            continue

        # if  sentence[KEY_sent] =='Johnson\'s new TV show, ``The Magic Hour,\'\' is just one aspect of a busy life:  -- HIS HEALTH: While by no means cured, he owes the appearance of remarkable health to a Spartan lifestyle and modern medicine.':
        #     print('here')
        scus = oie['verbs']
        in_sentence_scu_dict = {}
        tokens = oie['words']
        for scu in scus:
            tags = scu['tags']
            words = []
            if not ("B-ARG1" in tags or "B-ARG2" in tags or "B-ARG0" in tags):
                continue
            sub_scu_offsets = []
            scu_start_offset = None
            offset = 0
            initialSpace = 0
            while sentence[KEY_sent][offset + initialSpace] == ' ':
                initialSpace += 1  ## add space if exists, so 'offset' would start from next token and not from space
            offset += initialSpace
            for ind, tag in enumerate(tags):
                # if "ARG0" in tag or "ARG1" in tag or "V" in tag:
                assert (sentence[KEY_sent][offset] == tokens[ind][0])
                if "O" not in tag:
                    if scu_start_offset is None:
                        scu_start_offset = sentence[KEY_sent_char_idx] + offset

                        assert(sentence[KEY_sent][offset] == tokens[ind][0])

                    words.append(tokens[ind])
                else:
                    if scu_start_offset is not None:
                        spaceBeforeToken = 0
                        while sentence[KEY_sent][offset-1-spaceBeforeToken] == ' ':
                            spaceBeforeToken += 1## add space if exists
                        if sentence[KEY_sent][offset] == '.' or sentence[KEY_sent][offset] == '?':
                            dotAfter = 1 + spaceAfterToken
                            dotTest = 1
                        else:
                            dotAfter = 0
                            dotTest = 0
                        scu_end_offset = sentence[KEY_sent_char_idx] + offset - spaceBeforeToken + dotAfter

                        if dotTest:
                            assert (sentence[KEY_sent][offset - spaceBeforeToken + dotAfter -1] == tokens[ind-1+ dotTest][0]) #check only the dot, the start of the token
                        else:
                            assert (sentence[KEY_sent][offset - spaceBeforeToken + dotAfter - 1] == tokens[ind - 1 + dotTest][-1])  #check end of token
                        sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                        scu_start_offset = None


                ## update offset

                offset += len(tokens[ind])
                if ind < len(tags) - 1: #if not last token
                    spaceAfterToken = 0
                    while sentence[KEY_sent][offset + spaceAfterToken] == ' ':
                        spaceAfterToken += 1## add space after token if exists, so 'offset' would start from next token and not from space
                    offset += spaceAfterToken

            if scu_start_offset is not None: #end of sentence
                scu_end_offset = sentence[KEY_sent_char_idx] + offset
                sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                scu_start_offset = None



            # if len(words) <= 3:
            #     continue
            scuText = "...".join([sentence[KEY_sent][strt_end_indx[0] - sentence[KEY_sent_char_idx]:strt_end_indx[1] - sentence[KEY_sent_char_idx]] for strt_end_indx in sub_scu_offsets])
            #assert(scuText==" ".join([sentence[KEY_sent][strt_end_indx[0]:strt_end_indx[1]] for strt_end_indx in sub_scu_offsets]))
            in_sentence_scu_dict[scuText] = sub_scu_offsets

        notContainedDict = checkContained(in_sentence_scu_dict, sentence[KEY_sent], sentence[KEY_sent_char_idx])


        for scuText, binaryNotContained in notContainedDict.items():
            scu_offsets = in_sentence_scu_dict[scuText]
            if binaryNotContained:
                tmp = copy.deepcopy(sentence)
                tmp[KEY_scu_text] = scuText
                tmp[KEY_scu_offset] = scu_offsets
                scu_list.append(tmp)
    # select the best SCU
    # sort SCUs based on their length and select middle one
    # scu_list = sorted(scu_list, key=lambda x: len(x[KEY_scu_text].split()), reverse=True)
    # print(f"Best SCU:::{scu_list[int(len(scu_list)/2)]}")
    # return scu_list[int(len(scu_list)/2)]
    return scu_list


def read_and_split_sents(dataset):


    ## process all the documents files


    doc_sents = []
    for topic_dir in os.listdir(data_path):
        print(topic_dir)
        if topic_dir == 'summaries':
            continue

        topic = topic_dir.split('.')[0]

        topic_path = os.path.join(data_path, topic_dir)

        doc_files = os.listdir(topic_path)
        for doc_id in doc_files:
            document = read_generic_file(os.path.join(topic_path, doc_id))
            dsents = []
            # for line in document:
            #     dsents.extend(tokenize.sent_tokenize(line))
            dsents = tokenize.sent_tokenize(" ".join(document))
            idx_start = 0
            for dsent in dsents:
                if dsent != "...":  # this is a exception
                    doc_sents.append({'database': dataset, 'topic': topic, 'documentFile': doc_id, 'docSentCharIdx': idx_start,
                                      'docSentText': dsent})

                idx_start = idx_start + len(dsent) + 1  # 1 for the space charater between sentences


    return doc_sents



##################################
######     main     ##############
##################################
if __name__ == "__main__":
    DATASET = 'TAC2011'
    data_path = 'data/{}/'.format(DATASET)
    output_file = 'OIE_cands/OIE_cands_{}.csv'.format(DATASET)




    doc_spans = []

    doc_sents = read_and_split_sents(DATASET)
    doc_spans.extend(generate_scu_oie_multiSent(doc_sents, doc_summ='doc'))

    alignment_database_list = []
    for doc_span in doc_spans:
        doc_offset_str = offset_list2str(doc_span['docScuOffsets'])

        alignment_database_list.append([doc_span['database'],
                                             doc_span['topic'],
                                             doc_span['documentFile'],
                                             doc_span['docSentCharIdx'],
                                             doc_span['docSentText'],
                                             offset_list2str(
                                                 doc_span['docScuOffsets']),
                                             doc_span['docScuText']])



    alignment_database = pd.DataFrame(alignment_database_list,
                                           columns=['database', 'topic',
                                                    'documentFile',
                                                    'docSentCharIdx',
                                                    'docSentText', 'docSpanOffsets',
                                                    'docSpanText'])

    alignment_database.to_csv(output_file, index=False)

