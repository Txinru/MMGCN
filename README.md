# MMGCN
This code is the implementation of MMGCN

MMGCN：Multi-view Multichannel Attention Graph Convolutional Network for miRNA-disease Association Prediction

#### Requirements

python (tested on version 3.7)
pytorch (tested on version 1.5.1)
torch-geometric (tested on version 1.6.0)
numpy (tested on version 1.18.5)

#### Quick start

To reproduce our results:
Run main.py to RUN MMGCN.

#### Data description

d_d_f.csv: target-based similarity matrix for disease
d_d_s.csv: disease semantic similarity matrix
dis_name.csv: list of disease names.
m_d.csv: miRNA-disease association matrix
m_m_f.csv: miRNA functional similatiry matrix
m_m_s.csv: miRNA  sequence similarity matrix
miR_name.csv: list of miRNA names

#### Contacts

If you have any questions or comments, please feel free to email Xinru Tang(xinru@hnu.edu.cn) and/or Jiawei Luo( luojiawei@hnu.edu.cn)

