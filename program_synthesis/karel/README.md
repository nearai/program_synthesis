# Karel
Download the preprocessed dataset:
```
cd data
wget https://s3.us-east-2.amazonaws.com/karel-dataset/karel.tar.gz
tar xf karel.tar.gz
```

Train a model with SGD from `program_synthesis/karel` folder:
```
python train.py --dataset karel --model_type karel-lgrl \
  --debug_every_n=10000 --eval_every_n=10000 --keep_every_n=50000 \
  --log_interval=1000 --batch_size 128 --num_epochs 100 --max_beam_trees 1 \
  --optimizer sgd --gradient-clip 1 --lr 1 --lr_decay_steps 100000 --lr_decay_rate 0.5 \
  --model_dir logdirs/karel-sgd-cl1-lr1-lds100k-ldr0.5
```

Evaluate on the validation set:
```
python eval.py --model_type=karel-lgrl --dataset karel --max_beam_trees 64 --max_eval_trials 1 \
  --model_dir logdirs/karel-sgd-cl1-lr1-lds100k-ldr0.5
```
