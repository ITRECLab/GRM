<!-- English | [简体中文](./README.zh.md) -->
# GRM:Gaussian-enhanced Representation Model for Extracting Protein-Protein Interactions Affected by Mutations 
The implementation of method in our paper:
Extracting Protein-Protein Interactions Affected by Mutations via Gaussian-enhanced Representation and Contrastive Learning
## 1.Prepareration
### requirements
requirements are listed in `requirements.txt`
### DataSet & Evaluation
  Original dataset and evaluation scripts can be downloaded [here][bc6pm],

  And some of annotation of genes are modified as shown in <a href="#modi">the end of the document</a> 
  
  The modified datasets and the results obtained by GNP are available [here][dataset_googledrive].

## 2.Reproducing Results
### 10-fold cross validation on train
RC-Only: ` bash scripts/train_RC.sh`


### Train and eval on test
RC-Only: ` bash scripts/test_RC.sh`


The result of RC with confidence will be saved as `./outputjson/{loggercomment}_{epoch}.json`.

### Analysis
1. `analysis.ipynb`
   
    Predicted relations need to be post-processed here before homolo eval and exact eval.

    Scripts about case study can be found here. 
2. `python cross_fold_metrics.py > metrics.tsv`
    print results of the cross fold validation to `metrics.tsv`


[bc6pm]: https://github.com/ncbi-nlp/BC6PM 
[dataset_googledrive]: https://drive.google.com/file/d/17MCutWfCWA2rKpPnFp6gEJATdh-IYkZX/view?usp=sharing 
[workshop]: https://biocreative.bioinformatics.udel.edu/resources/publications/bcvi-proceedings/
[eutil]: https://pypi.org/project/eutils/
[biobert]: https://github.com/dmis-lab/biobert
[bertviz]: https://github.com/jessevig/bertviz
