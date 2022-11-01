## About The Repository:
This repository contains the code and data for the following paper:\
 <a href="https://aclanthology.org/2020.coling-main.197.pdf"> Style versus Content: A distinction without a (learnable) difference? </a> \
The models introduced in this paper are extensions of the model  <a href="https://github.com/shentianxiao/language-style-transfer"> </a>  introduced by the paper <a href="https://arxiv.org/pdf/1705.09655v2.pdf">《Style Transfer from Non-Parallel Text by Cross-Alignment》</a> 

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up of the repository and run the model follow these steps.
<!--*****************************my comments -->
#### Requirements 
* torchtext >= 0.4.0 ????
* nltk ??????
* fasttext == 0.9.3?????
* kenlm ?????

#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/somayeJ/RNN-based-TST-experiments.git
   ```
2. Install the requirements
<!--*************************1.notes to myself -->
#### Running the model
* To run the model, first adjust the following parameters in the of the file  ''options.py''

* Then run the following command:
   ```sh
   python model.py
   ```
## Data 
* The data/Yelp/ directory contains the  Yelp restaurant reviews dataset used in the paper <a href="https://arxiv.org/abs/1705.09655">《Style Transfer from Non-Parallel Text by Cross-Alignmen》</a>. 
*  Data format: Each file should consist of one sentence per line with tokens separated by a space. The two styles are represented by 0 and 1

## Dependencies
* pytorch = 
* Python = 
