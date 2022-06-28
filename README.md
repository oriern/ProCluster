# ProCluster
Code and Model for "Proposition-Level Clustering for Multi-Document Summarization" paper


--***This repository is still under construction***--


`supervised_oie_wrapper` directory is a wrapper over AllenNLP's (v0.9.0) pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.


## How to generate summaries? ##

  1. Download the trained models from [here](https://drive.google.com/file/d/1CNaTH1k5oflmGiljQ7JL6NQ_3uz5tdvq/view?usp=sharing), and put them in 'models' directory.
  2. Put your data in `data\<DATASET>\` directory. (For example `data\DUC2004\`)
  3. Setup Huggingface Transformers repository (v4.2.2):
   
     a. 
      ```
        git clone https://github.com/huggingface/transformers
        git checkout v4.2.2-patch
      ```
     
     b. Move `transformers\modeling_longformer.py` from this repo to the new transformers repo: `transformers\src\transformers\models\longformer\modeling_longformer.py`
     
     c. Move `transformers\run_glue_highlighter.py` from this repo to the new transformers repo: `transformers\examples\text-classification\run_glue_highlighter.py`
     
     d. 
      ```
        cd transformers
        pip install .
      ```
  
  3. Extract all Open Information Extraction (OIE) spans from the source documents:
  ```
    python extract_OIEs.py
  ```
  4. Prepare the data for the Salience model:
  ```
    python DataGenSalientIU_DUC_allAlignments_CDLM.py
  ```
  5. Predict salience score for each OIE span:
  ```
    python extract_OIEs.py
  ```
 
  **[Optional]** 5*. Cluster salient spans, rank clusters, and select the most salient span to represent each cluster:
    ("Salience_prop + Clustering" model in Sec 4.3 in the paper)
   ```
    python deriveSummaryDUC.py
  ```
  
  6. Cluster salient spans and prepare data for the Fusion model:
   ```
    python prepare_fusion_data.py
  ```
  7. Generate a fused sentence from every cluster:
   ```
    python deriveSummaryDUC_fusion_clusters.py
  ```
  8. Concatinate the fused sentences, and calculate final ROUGE scores:
   ```
    python deriveSummaryDUC_fusion_clusters.py
  ```
