# import spacy
# from supervised_oie_wrapper.run_oie import run_oie
# from allennlp.predictors.predictor import Predictor
# from allennlp.predictors.open_information_extraction import make_oie_string, get_predicate_text, OpenIePredictor
# import argparse
from utils import *
from Aligner import Aligner

# from BaseAligner import BaseAligner
# from RougeAligner import RougeAligner
# from RougeHighlighter import RougeHighlighter
# from ClozeAligner import ClozeAligner
# from annotation2MRPC_Aligner import annotation2MRPC_Aligner
# from cnn_dmAligner import cnn_dmAligner
# import pandas as pd
sys.path.append('/home/nlp/ernstor1/transformers/examples/')
sys.path.append('/home/nlp/ernstor1/autoAlignment/SCUdataGenerator/')
from inDoc2MRPC_Aligner import inDoc2MRPC_Aligner

import run_glue

import contextlib
@contextlib.contextmanager
def redirect_argv(num):
    sys._argv = sys.argv[:]
    sys.argv = str(num).split()
    yield
    sys.argv = sys._argv






parser = argparse.ArgumentParser()
parser.add_argument('-data_path', type=str, default='/home/nlp/ernstor1/data/TAC2008/train/')#'/home/nlp/ernstor1/DUC2004/')  # 'data/final_data/data')
parser.add_argument('-mode', type=str, default='dev')
parser.add_argument('-input_file_path', type=str, default='data/final_data/fullAlignmentDataset_dev_MACE.csv')
parser.add_argument('-log_file', type=str, default='results/dev_log.txt')
# parser.add_argument('-metric_precompute', type=str2bool, default=False)
parser.add_argument('-output_file', type=str, default='/home/nlp/ernstor1/transformers/data/newMRPC_OIU/devTmp/dev.tsv')
parser.add_argument('-database', type=str, default='tac2008')#,duc2007,MultiNews')
# parser.add_argument('-intermidiate_steps_to_csv', type=str2bool, default=False)
args = parser.parse_args()




aligner = inDoc2MRPC_Aligner(data_path=args.data_path, mode=args.mode, input_file_path=args.input_file_path,
                 log_file=args.log_file, output_file = args.output_file,
                 database=args.database)
logging.info(f'output_file_name: {args.output_file}')

topic_dirs = glob.glob(f"{args.data_path}/*")
for topic_dir in topic_dirs[-2:]:
        if topic_dir.split('/')[-1] == 'summaries':
            continue


        print ('Starting with summary {}'.format(topic_dir))
        aligner.new_topic_init()
        aligner.read_and_split(topic_dir)
        aligner.scu_span_aligner()
        aligner.save_predictions()
        with redirect_argv('python --model_type roberta --model_name_or_path roberta-large-mnli --task_name MRPC --do_eval'
                           ' --calc_alignment_sim_mat --weight_decay 0.1 --data_dir /home/nlp/ernstor1/transformers/data/newMRPC_OIU/devTmp/'
                           ' --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 2e-6'
                           ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
                           ' --output_dir /home/nlp/ernstor1/transformers/examples/out/outnewMRPC_OIU/SpansOieNegativeAll_pan_full089_fixed/checkpoint-2000/'):
            run_glue.main()



        # os.system('/home/nlp/ernstor1/transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-large-mnli --task_name MRPC --do_eval'
        #   ' --calc_alignment_sim_mat --weight_decay 0.1 --data_dir /home/nlp/ernstor1/transformers/data/newMRPC_OIU/devTmp/'
        #   ' --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 2e-6'
        #   ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
        #   ' --output_dir /home/nlp/ernstor1/transformers/examples/out/outnewMRPC_OIU/SpansOieNegativeAll_pan_full089_fixed/checkpoint-2000/')




