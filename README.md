RED Server Software
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.

# red-paper
Code and supporting materials for the AAAI 2022 RED paper

This repository contains all the source codes to reproduce the experimental results reported in paper "Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model", which is published in AAAI 2022. (Arxiv Link: https://arxiv.org/abs/2010.02065)

## Steps to reproduce the main results in the paper:

1. Before running the source code, download the 121 UCI datasets from https://github.com/bioinf-jku/SNNs, and the 4 newly added datasets from UCI Machine Learning Repository, namely, "Phishing","messidor","Bioconcentration", and "Climate".
2. Put the 121 UCI datasets into a folder named ```UCI121_data```, and the 4 newly added dataset into a folder named ```Datasets```. 
3. Create a folder named ```Results``` to store all the experimental results.
4. Create a folder named ```Statistics``` to store all the analysis results.
5. pip install the packages in ```requirements.txt```.
6. Run ```main_experiments_UCI.py``` to generate all the experimental results.
7. Run ```analysis_results_UCI.py``` to analyze the stored experimental results. 

Note: the packages used in calculating Trust Score can be git cloned from https://github.com/google/TrustScore.

## Steps to reproduce other experimental results in the paper:

1. For results of "SVGP" variant, run ```experiments_SVGP.py``` after obtaining the main results in ```Results``` folder.
2. For results of "DNGO" variant, run ```experiments_DNGO.py``` after obtaining the main results in ```Results``` folder. The source codes of DNGO can be git cloned from https://github.com/automl/pybnn/blob/master/pybnn/dngo.py. 
3. For results of "Entropy" variant, it can be directly calculated from the outputs of original NN classifier using ```entropy``` function from ```scipy.stats```.
4. For results regarding Bayesian Neural Networks, run ```experiments_BNN.py``` to generate the results of base BNN classifier, then run ```experiments_BNN+RED.py``` to get the results of applying RED on top of BNN classifiers. 
5. For results regarding MC-dropout, run ```experiments_dropout.py``` to generate the results of base NN classifier with MC-dropout, then run ```experiments_dropout+RED.py``` to get the results of applying RED on top of NN classifiers with MC-dropout. 
6. For results regarding VGG16/VGG19 on CIFAR-10/CIFAR-100, train the base VGG16/VGG19 classifiers using code from https://github.com/geifmany/cifar-vgg, then run the included codes that generate results on UCI datasets on CIFAR-10/CIFAR-100 results.
7. For results regarding OOD and adversarial samples, run ```experiments_OOD_adversarial.py```.
8. For results regarding OOD detection in CIFAR-10 vs. SVHN, directly apply the trained VGG16 and RED models in step 6 to SVHN data downloaded from http://ufldl.stanford.edu/housenumbers/.
9. For results of BLR-residual, run ```experiments_BLR-residual.py``` after running ```main_experiments_UCI.py```.

## Citation

If you use RED in your research, please cite it using the following BibTeX entry
```
@misc{Qiu2022RED,
      title={Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model}, 
      author={Xin Qiu and Risto Miikkulainen},
      year={2022},
      eprint={2010.02065},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
