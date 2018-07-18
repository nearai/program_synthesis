Training script:

```bash
python train.py --model_dir=/home/ubuntu/models/seq2seq \
--log_interval=100 --num_placeholders=50 --bidirectional --num_steps=25000 \
--num_units=500 --max_decoder_length=500 --optimizer=adam --clip=0.5 \
--lr=0.00025 --lr_decay_steps=5000 --lr_decay_rate=0.5 --keep_every_n=5000 \
--optimizer_weight_decay=0 --num_encoder_layers=2 \
--dataset_filter_code_length=500 --num_att_heads=1 \
--decoder_dropout=0.2 --encoder_dropout=0.2 --batch_size=100
```

Eval script:

```bash
python eval.py --model_dir=/home/ubuntu/models/seq2seq \
--max_beam_trees=64 --max_eval_trials=64 --batch_size=10 \
--max_decoder_length=1000 --num_placeholders=50 
```
