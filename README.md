# Omniglot
## dataset for few shot learning
Refer from repo: https://github.com/cnielly/prototypical-networks-omniglot

Data train download:
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
Data test download:
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip

The Omniglot dataset is taken on the official GitHub repository: https://github.com/brendenlake/omniglot

## Environment
Pytroch 1.7, python 3.8
Run to train and test: 
CUDA_VISIBLE_DEVICES=0 python protonet.py
