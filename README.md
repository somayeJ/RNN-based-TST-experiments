## About The Repository:
This repository contains the code and data for the following paper:\
 <a href="https://aclanthology.org/2020.coling-main.197.pdf"> Style versus Content: A distinction without a (learnable) difference? </a> \
The models introduced in this paper are extensions of the  <a href="https://github.com/shentianxiao/language-style-transfer"> model  </a>  introduced by the paper <a href="https://arxiv.org/pdf/1705.09655v2.pdf"> Style Transfer from Non-Parallel Text by Cross-Alignment </a> 

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up of the repository and run the model follow these steps.
<!--*****************************my comments -->
<!--***************************** #### Requirements 
* torchtext >= 0.4.0 ????
* nltk ??????
* fasttext == 0.9.3?????
* kenlm ?????  my comments -->

### Installation

Clone the repo
   ```sh
   git clone https://github.com/somayeJ/RNN-based-TST-experiments.git
   ```
<!--*****************************my comments -->
<!--***************************** 2. Install the requirements?  my comments -->
<!--*************************1.notes to myself -->
### Running the codes
####  TST models
* To run the models, first adjust the parameters in the function of "load_arguments()" of the file "model.py"

* Then run the following command:
   ```sh
   python model.py
   ```
####  Evaluation
* Content Preservation Power
    * Download  the  glove embeddings (put it in codes/evaluation/content_preservation_power/embedding_based/)
    * Adjust the parameters in the function of "load_arguments()" of the code
* Style Shift Power (SSP)
    * Train the classifier on the gold data  or test the SSP of the model outputs by:
    * Adjusting the parameters in the function of "load_arguments()" of the code
    * Then running the following command:
   ```sh
   python classifier.py
   ```
* Fluency
    * Train the language model on the gold data  or test the fluency of the model outputs by:
    * Adjusting the parameters in the function of "load_arguments()" of the code
    * Then running the following command:
   ```sh
   python language_model.py
   ```
## Data 
* The data/Yelp/ directory contains the  Yelp restaurant reviews dataset used in the paper <a href="https://arxiv.org/abs/1705.09655">《Style Transfer from Non-Parallel Text by Cross-Alignmen》</a>. 
*  Data format: Each file should consist of one sentence per line with tokens separated by a space. The two styles are represented by 0 and 1

## Dependencies
* TensorFlow 1.3.0 
* Python >= 2.7 
