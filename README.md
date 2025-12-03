# Welcome to: A two-stage computational framework for identifying antiviral peptides and their functional types based on contrastive learning and multi-feature fusion strategy
Antiviral peptides (AVPs) have shown potential in inhibiting viral attachment, preventing viral fusion with host cells and disrupting viral replication due to their unique action mechanisms. They have now become a broad-spectrum, promising antiviral therapy. However, identifying effective AVPs is traditionally slow and costly. This study proposed a new two-stage computational framework for AVP identification. The first stage identifies AVPs from a wide range of peptides, and the second stage recognises AVPs targeting specific families or viruses. This method integrates contrastive learning and multi-feature fusion strategy, focusing on sequence information and peptide characteristics, significantly enhancing predictive ability and interpretability. The evaluation results of the model show excellent performance, with accuracy of 0.9240 and MCC score of 0.8482 on the non-AVP independent dataset, and accuracy of 0.9934 and MCC score of 0.9869 on the non-AMP independent dataset. Furthermore, our model can predict antiviral activities of AVPs against six key viral families (Coronaviridae, Retroviridae, Herpesviridae, Paramyxoviridae, Orthomyxoviridae, Flaviviridae) and eight viruses (FIV, HCV, HIV, HPIV3, HSV1, INFVA, RSV, SARS-CoV).

This AVP prediction tool developed by a team from the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study](https://github.com/GGCL7/CAVP/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Datasets](https://github.com/GGCL7/CAVP/tree/main/Datasets)

## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/GGCL7/AVP-IFT.git
cd AVP-IFT
```
2. **Set up the Python environment**
```bash
conda create -n avpift python=3.10
conda activate avpift
pip install -r requirements.txt
```

# Model source code
The source code for training our models can be found here [Model source code](https://github.com/GGCL7/AVP-IFT/tree/main/Code).


