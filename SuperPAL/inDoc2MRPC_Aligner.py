from Aligner import Aligner
from annotation2MRPC_Aligner import annotation2MRPC_Aligner
from utils import *
import pandas as pd
import pickle



class inDoc2MRPC_Aligner(annotation2MRPC_Aligner):
    # Parse document sentences to OIE and prepare all OIE pairs combinations for supervised aligner

    def __init__(self, data_path='.', mode='dev', input_file_path='data/final_data/fullAlignmentDataset_dev_MACE.csv',
                 log_file='results/dev_log.txt', output_file = './dev.tsv',
                 database='duc2004,duc2007,MultiNews'):
        super().__init__(data_path=data_path, mode=mode, input_file_path=input_file_path,
                 log_file=log_file, metric_precompute=False, output_file = output_file,
                 database=database)

        self.alignment_database_list = []
        self.idx2span = {}

    def new_topic_init(self):
        self.alignment_database_list = []
        self.idx2span = {}
        self.alignment_database = None
        self.doc_sents = []


    def main_filter(self, doc_span1, doc_spans, doc_span1_idx):



        scu_offset_str = offset_list2str(doc_span1['docScuOffsets'])
        id_doc_sent1 = doc_span1['topic'] + '_' + scu_offset_str

        for doc_span2_idx, doc_span2 in enumerate(doc_spans):
            doc_offset_str = offset_list2str(doc_span2['docScuOffsets'])
            id_doc_sent2 = doc_span1['topic'] + '_' + doc_span2['documentFile'] + '_' + doc_offset_str
            label = 0 #label =0 for all. positive samples' label would be changed later

            self.alignment_database_list.append([label, id_doc_sent1, id_doc_sent2,
                                                 doc_span1['docScuText'],
                                                 doc_span2['docScuText'],
                                                 doc_span2['topic'], doc_span1['documentFile'],
                                                 doc_span1['docSentCharIdx'],
                                                 doc_span1['docSentText'],
                                                 doc_span2['documentFile'],
                                                 doc_span2['docSentCharIdx'],
                                                 doc_span2['docSentText'],
                                                 offset_list2str(
                                                     doc_span2['docScuOffsets']),
                                                 offset_list2str(doc_span1['docScuOffsets']),
                                                 doc_span2['docScuText'], doc_span1['docScuText'],
                                                 str(doc_span1_idx) +','+str(doc_span2_idx)])



    def read_and_split(self, topic_dir):

        ## process all the documents files
        doc_files = glob.glob(f"{topic_dir}/*")

        logging.info(f"Following documents have been found for them:")
        logging.info("\n".join(doc_files))
        self.doc_sents = []
        for df in doc_files:
            doc_id = os.path.basename(df)
            document = read_generic_file(df)
            dsents = tokenize.sent_tokenize(" ".join(document))
            idx_start = 0
            for dsent in dsents:
                if dsent != "...":  # this is a exception
                    self.doc_sents.append({'topic':os.path.basename(topic_dir), 'documentFile': doc_id, 'docSentCharIdx': idx_start,
                                      'docSentText': dsent})

                idx_start = idx_start + len(dsent) + 1  # 1 for the space charater between sentences






    def scu_span_aligner(self):
        """ Module which align scu and sentence
        in the document given a summary and document
        """

        doc_spans = []
        doc_spans.extend(generate_scu_oie_multiSent(self.doc_sents, doc_summ='doc'))

        for doc_span1_idx, doc_span1 in enumerate(doc_spans):
            self.idx2span[doc_span1_idx] = doc_span1
            self.main_filter(doc_span1, doc_spans, doc_span1_idx)











    def save_predictions(self):

        self.alignment_database = pd.DataFrame(self.alignment_database_list,
                                               columns=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String',
                                                         'topic',
                                                        'summaryFile', 'scuSentCharIdx', 'scuSentence', 'documentFile',
                                                        'docSentCharIdx',
                                                        'docSentText', 'docSpanOffsets', 'summarySpanOffsets',
                                                        'docSpanText', 'summarySpanText', 'sim_mat_idx'])
        self.alignment_database.to_csv(self.output_file, index=False, sep='\t')

        with open('/home/nlp/ernstor1/main_summarization/sim_mats/{}'.format(self.alignment_database['topic'].iloc[0]
                                                                             +'_idx2span.pickle'), 'wb') as handle:
            pickle.dump(self.idx2span, handle)



