# ProCluster
Code and Model for "Proposition-Level Clustering for Multi-Document Summarization" paper


--***This repository is still under construction***--


`supervised_oie_wrapper` directory is a wrapper over AllenNLP's (v0.9.0) pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.

You are welcome to try our [demo](https://studio.oneai.com/). Look for the `Multi-Doc Summary by Ernst et al` skill.

## How to generate summaries? ##

### Preliminary steps ###

  1. Download the trained models from [here](https://drive.google.com/file/d/1CNaTH1k5oflmGiljQ7JL6NQ_3uz5tdvq/view?usp=sharing), and put them in 'models' directory.
  2. Put your data in `data\<DATASET>\` directory. (For example `data\DUC2004\`)
  3. Install `requirements.txt` (python 3.6)
   4. Create similarity matrix by SuperPAL (to be used for the clustering step):
     
      a. Clone [SuperPAL](https://github.com/oriern/SuperPAL) repository.
     
      b. Move files from `SuperPAL` folder in this repository to the new `SuperPAL` repository.
     
      c. Follow the steps that appear in SuperPAL repository under 'Alignment model' section.
         
         &emsp;Instead of step 2, run:
         
         ```
          python main_predict_inDoc.py -data_path <DATA_PATH>  -output_path <OUT_DIR_PATH>  -alignment_model_path  <ALIGNMENT_MODEL_PATH>
         ```
   
   **[Optional]** 5. Follow [this](https://github.com/OriShapira/SummEval_referenceSubsets) repository to install the official ROUGE measure. 
  
  ### Generating summaries ###
  
  1. Extract all Open Information Extraction (OIE) spans from the source documents:
  ```
    python extract_OIEs.py
  ```
  2. Prepare the data for the Salience model:
  ```
    python DataGenSalientIU_DUC_allAlignments_CDLM.py
  ```
  3. Predict salience score for each OIE span:
  ```
     cd transformers\examples\text_classification\
     python run_glue_highlighter.py --model_name_or_path <MODEL_PATH>  --train_file <DATA_CSV_FILE_PATH> --validation_file <DATA_CSV_FILE_PATH>   --do_predict   --evaluation_strategy steps --eval_steps 250 --save_steps 250 --max_seq_length 4096 --gradient_accumulation_steps 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-5 --num_train_epochs 3 --output_dir <OUTPUT_DIR>
  ```
 
  **[Optional]** 3*. Cluster salient spans, rank clusters, and select the most salient span to represent each cluster:
    ("Salience_prop + Clustering" model in Sec 4.3 in the paper)
   ```
    python deriveSummaryDUC.py
  ```
  
  4. Cluster salient spans and prepare data for the Fusion model:
   ```
    python prepare_fusion_data.py
  ```
  5. Generate a fused sentence from every cluster:
   ```
    cd transformers\examples\seq2seq\
    python finetune_trainer.py --model_name_or_path=<MODEL_PATH> --learning_rate=3e-5  --do_predict --num_train_epochs=4 --evaluation_strategy steps --predict_with_generate --eval_steps=50 --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --max_source_length=265 --eval_beams=6 --max_target_length=30 --val_max_target_length=30 --test_max_target_length=30 --data_dir <DATA_CSV_FILE_PATH> --output_dir <OUTPUT_DIR>
  ```
  6. Concatinate the fused sentences, and calculate final ROUGE scores:
   ```
    python deriveSummaryDUC_fusion_clusters.py
  ```
