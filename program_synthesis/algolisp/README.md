# AlgoLisp

This work is described in:
* Paper: [Neural Program Search: Solving Programming Tasks from Description and Examples](https://arxiv.org/abs/1802.04335) (https://arxiv.org/abs/1802.04335)
* Blog post: http://near.ai/articles/2018-05-07-Neural-Program-Search-Paper/

## Download the dataset

The metaset3 dataset is available here:

 - Train: https://www.dropbox.com/s/qhun6kml9yb2ui9/metaset3.train.jsonl.gz
 - Dev: https://www.dropbox.com/s/aajkw83j2ps8bzx/metaset3.dev.jsonl.gz
 - Test: https://www.dropbox.com/s/f1x9ybkjpf371cp/metaset3.test.jsonl.gz

Download and unpack data into `../../data/generated` folder

## Train models

An easy way to start training model:

    python train.py --model_type=seq2seq --model_dir=models/seq2seq
