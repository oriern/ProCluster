
from utils import *
from Aligner import Aligner
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
parser.add_argument('-data_path', type=str, required=True)
parser.add_argument('-mode', type=str, default='dev')
parser.add_argument('-log_file', type=str, default='results/dev_log.txt')
parser.add_argument('-output_path', type=str, required=True)
parser.add_argument('-alignment_model_path', type=str, required=True)
parser.add_argument('-database', type=str, default='None')
args = parser.parse_args()




aligner = inDoc2MRPC_Aligner(data_path=args.data_path, mode=args.mode,
                 log_file=args.log_file, output_file = args.output_path,
                 database=args.database)
logging.info(f'output_file_path: {args.output_path}')

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
                           ' --calc_alignment_sim_mat --weight_decay 0.1 --data_dir {args.output_path}'
                           ' --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 2e-6'
                           ' --logging_steps 500 --num_train_epochs 2.0 --evaluate_during_training  --overwrite_cache'
                           ' --output_dir {args.alignment_model_path}'):
            run_glue.main()




