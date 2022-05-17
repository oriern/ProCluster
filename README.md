# ProCluster
Code and Model for "Proposition-Level Clustering for Multi-Document Summarization" paper


`supervised_oie_wrapper` directory is a wrapper over AllenNLP's (v0.9.0) pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.

You can use `py36.yml` to restore all requirements.

## How to generate summaries? ##

  1. Download the trained models from [here](), and put them in 'models' directory.
  2. Extract all Open Information Extraction (OIE) spans from the source documents:
    
    ```
      python extract_OIEs.py
    ```
