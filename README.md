# CoraL

### Introduction

This repository contains code for the paper "CoraL: Interpretable contrastive meta-learning for the prediction of cancer-associated ncRNA-encoded small peptides".

NcRNA-encoded small peptides (ncPEPs) have recently emerged as promising targets and biomarkers for cancer immunotherapy. Therefore, identifying cancer-associated ncPEPs is crucial for cancer research. In this work, we propose CoraL, a novel supervised contrastive meta-learning framework for predicting cancer-associated ncPEPs. Specifically, the proposed meta-learning strategy enables our model to learn meta-knowledge from different types of peptides and train a promising predictive model even with few labelled samples. The results show that our model is capable of making high-confidence predictions on unseen cancer biomarkers with five samples only, potentially accelerating the discovery of novel cancer biomarkers for immunotherapy. Moreover, our approach remarkably outperforms existing deep learning models on 15 cancer-associated ncPEPs datasets, demonstrating its effectiveness and robustness. Interestingly, our model exhibits outstanding performance when extended for the identification of short open reading frames (sORFs) derived from ncPEPs, demonstrating the strong prediction ability of CoraL at the transcriptome level. Importantly, our feature interpretation analysis discovers unique sequential patterns as fingerprint for each cancer-associated ncPEPs, revealing the relationship among certain cancer biomarkers that are validated by relevant literature and motif comparison. Overall, we expect CoraL to be a useful tool to decipher the pathogenesis of cancer and provide valuable information for cancer research.

### Acknowledgement

Thanks to Wenjia He (He used to be a member of Weilab and now continues his PhD life in the King Abdullah University of Science and Technology). He provided some advice and guidance on building the CoraL framework.
