import glob, re, os
for logdir in glob.glob(
        'logdirs/20180211/karel-lgrl-ref-edit-m12-sgd-cl1-lr0.1-lds100k-ldr0.5'):
    for ckpt in sorted(glob.glob(logdir + '/checkpoint-????????')):
        ckpt_number = int(ckpt[-8:])
        cm100 = ckpt_number - 100
        if (cm100 % 50000 != 0 and cm100 % 10000 == 0) or ckpt_number < 1000:
            continue
        for dist in ('1', '0,1', '0,0,1'):
            print('python eval.py --model_type karel-lgrl-ref '
                  '--dataset karel --max_beam_trees 64 --step {step} '
                  '--karel-mutate-ref --karel-mutate-n-dist {dist} '
                  '--model_dir {logdir} '
                  '--report-path {logdir}/report-dev-m{dist}-{step}.jsonl '
                  '--hide-example-info').format(
                      step=ckpt_number, logdir=logdir, dist=dist)
