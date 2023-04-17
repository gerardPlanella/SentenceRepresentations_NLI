# SentenceRepresentations_NLI
This project aims is the first practical of the Advanced Topics in Computatonal Semantics subject of the Artificial Intelligence master's at the University of Amsterdam. The goal is to learn general general-purpose sentence representations through Natural Language Inference task, implementing four different neural models for sentence classification:
- Average Word Embeddings
- Unidirectional LSTM Encoder
- Bidirectional LSTM Encoder
- Bidirectional LSTM Encoder with Max pooling

After trainig these models on the Stanford Natural Language Inference (SNLI) corpus, we will evaluate the obtained sentence representations on Facebook's SentEval evaluation framework. Which will apply our obtained models to different unseen transfer tasks. The goal is to reproduce the results found in the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://aclanthology.org/D17-1070) (Conneau et al., EMNLP 2017).

## Code Structure
|-dataset/: This folder is where we will save the pre-trained embeddings, the dataset vocabulary file and the vocabulary files. It should start empty.
|-models/: This folder is where we will save the .pt checkpoint files for the different pretrained models, they can be found in the following Google drive folder.
|-results/: Directory containing the saved results from SentEval.
|-runs/:Directory containing Tensorboard logs for the SNLI training of the different models.
|-analysis.ipynb: Jupyter Notebook containing result visualization and analysis.
|-data.py: Contains definition of dataset, vocabulary, feature vectors and data loader functions
|-environment.yml: Environment CPU only file.
|-environmeny_gpu.yml: Environment for CUDA GPU enabled machined.
|-evaluation.py: Contains evaluation functions for SNLI and model inference functions.
|-models.py: Sentence Encoder and Classifier classes
|-run_awe.job: Job file for running AWE training on the universities' GPU cluster Lisa.
|-run_bilstm_max.job: Job file for running BiLSTM encoder with max pooling on Lisa.
|-run_bilstm.job: Job file for running BiLSTM encoder on Lisa.
|-run_change_dataset.job: Job file for updating dataset vocabulary files. Use it if you want to change dataset, tokenizer, or data percentage used. 
|-run_lstm_max.job: Job file for running Unidirectional LSTM encoder on Lisa.
|-run_senteval_awe.job: Job file for running SentEval evaluation for AWE encoder on Lisa.
|-run_senteval_bilstm_max.job: Job file for running SentEval evaluation for BiLSTM encoder with max pooling on Lisa.
|-run_senteval_bilstm.job: Job file for running SentEval evaluation for BiLSTM encoder on Lisa.
|-run_senteval_lstm.job: Job file for running SentEval evaluation for LSTM encoder on Lisa.
|-senteval.py: File used for model SentEval evaluation.
|-train_nli.py: File used for model training on SNLI.
|-train.py: File containing training functions used for training in train_nli.py.

## Getting Started
1. Download GloVe embeddings
```
        cd dataset
        wget -P /dataset/ -q https://nlp.stanford.edu/data/glove.840B.300d.zip
        cd ..
```
2. Install Environment
* For CPU Only:
```
        conda env create -f "environment.yml"
        conda activate ATCS
```
* For GPU:
```
        conda env create -f "environment_gpu.yml"
        conda activate ATCS_GPU
```
3. Download Pretrained models and Vocabulary files from #TODO: Add link
    * Copy the models, results folders and paste them to the repositorie's base folder.
    * Copy the contents of the folder and move it to the dataset folder of the repository.        
3. Clone SentEval repo into folder at same level as repositorie's
```
        git clone https://github.com/facebookresearch/SentEval.git
        cd SentEval/
```
4. Install SentEval
```
        python setup.py install
```
5. Download datasets for downstream tasks
```
        cd data/downstream/
        ./get_transfer_data.bash
```
6. Modify SentEval/senteval/utils.py file for Python versions >= 3.10
    In the function: 
```
        def get_optimizer(s):
```
    There is the following line:
```
        expected_args = inspect.getargspec(optim_fn.__init__)[0]
```
    Change it to this (Pull Request to facebookresearch pending approval):
```
        import sys
        if sys.version_info < (3, 10):
            expected_args = inspect.getargspec(optim_fn.__init__)[0]
        else:
            expected_args = list(inspect.signature(optim_fn.__init__).parameters.keys())
```

## Model Training
The scrpit used for training is the train_nli.py script. It can be run with the following arguments: 
```bash
usage: train_nli.py [-h] [--num_epochs NUM_EPOCHS] [--embedding_dim EMBEDDING_DIM] [--classifier_fc_dim CLASSIFIER_FC_DIM] [--lr LR] [--lr_decay LR_DECAY] [--encoder_dropout ENCODER_DROPOUT]
                    [--encoder_pooling ENCODER_POOLING] [--encoder_lstm_dim ENCODER_LSTM_DIM] [--encoder ENCODER] [--data_percentage DATA_PERCENTAGE] [--reload_dataset] [--dataset DATASET]
                    [--tokenizer TOKENIZER] [--dataset_vocab_path DATASET_VOCAB_PATH] [--vocab_path VOCAB_PATH] [--embedding_path EMBEDDING_PATH] [--checkpoint_path CHECKPOINT_PATH]
                    [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--tensorboard_dir TENSORBOARD_DIR] [--lr_factor LR_FACTOR] [--seed SEED]

NLI training

options:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
  --embedding_dim EMBEDDING_DIM
  --classifier_fc_dim CLASSIFIER_FC_DIM
  --lr LR
  --lr_decay LR_DECAY
  --encoder_dropout ENCODER_DROPOUT
  --encoder_pooling ENCODER_POOLING
  --encoder_lstm_dim ENCODER_LSTM_DIM
  --encoder ENCODER
  --data_percentage DATA_PERCENTAGE
  --reload_dataset
  --dataset DATASET
  --tokenizer TOKENIZER
  --dataset_vocab_path DATASET_VOCAB_PATH
  --vocab_path VOCAB_PATH
  --embedding_path EMBEDDING_PATH
  --checkpoint_path CHECKPOINT_PATH
  --batch_size BATCH_SIZE
  --eval_batch_size EVAL_BATCH_SIZE
  --tensorboard_dir TENSORBOARD_DIR
  --lr_factor LR_FACTOR
  --seed SEED           seed
```

There are many parameters that you can play around with but to reproduce the original paper's results most of the default values have already been set:

1. Average Word Embeddings:
```
        python train_nli.py --encoder "awe"
```
2. Unidirectional LSTM:
```
        python train_nli.py --encoder "lstm"
```
3. Bidirectional BiLSTM: 
```
        python train_nli.py --encoder "bilstm"
```
4. Bidirectional BiLSTM with Max Pooling:
```
        python train_nli.py --encoder "bilstm" --encoder_pooling "max"
```
After training, the models will be saved in the models/ directory by default.

## SNLI Evaluation
The train_nli.py script will also evaluate the model on SNLI after training, but to evaluate a model on the testing split after loading it from a checkpoint, one can use the provided [analysis.ipynb](analysis.ipynb) file, where this is performed for each model. 

### SentEval Evaluation
The script senteval.py will evaluate a model given a path to its checkpoint, these are the possible input arguments one can use:
```bash
usage: senteval.py [-h] [--data_path DATA_PATH] [--vocab_path VOCAB_PATH] [--model_path MODEL_PATH] [--embedding_path EMBEDDING_PATH] [--kfold KFOLD] [--tokenizer TOKENIZER] [--usepytorch]
                   [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--optim OPTIM] [--results_path RESULTS_PATH]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --vocab_path VOCAB_PATH
  --model_path MODEL_PATH
  --embedding_path EMBEDDING_PATH
  --kfold KFOLD
  --tokenizer TOKENIZER
  --usepytorch
  --batch_size BATCH_SIZE
  --num_epochs NUM_EPOCHS
  --optim OPTIM
  --results_path RESULTS_PATH
```
If you placed the SentEval repository at a level different to the repositorie's you will have to manually change the senteval_path variable found in the top of the senteval.py file to the path of the cloned SentEval repository. To run the evaluations most of the default parameters used Conneau et al.'s paper have already been set. The command used for generating the standard results is the following:
```
        python senteval.py --usepytorch --model_path <path_to model>
```
The evaluation results will be saved in the results/ folder by default and can be modified with the --results_path argument.

## Author
Gerard Planella Fontanillas
Email: gerard.plfo@gmail.com
Email2: gerard.planella.fontanillas@student.uva.nl

## Acknowledgements
* Code structure based on the [2022 Natural Language Processing 1 course](https://cl-illc.github.io/nlp1-2022/)'s pipeline.






