import glob, re, os
for (logdir, ckpt_number, dist) in [
        # token m1
        #('logdirs/20180201/karel-lgrl-ref-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #300100,
        #'1'),
        #('logdirs/20180201/karel-lgrl-ref-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #200100,
        #'0,1'),
        #('logdirs/20180201/karel-lgrl-ref-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #150100,
        #'0,0,1'),

        ## token m1,2
        #('logdirs/20180201/karel-lgrl-ref-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #200100,
        #'1'),
        #('logdirs/20180201/karel-lgrl-ref-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #300100,
        #'0,1'),
        #('logdirs/20180201/karel-lgrl-ref-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #150100,
        #'0,0,1'),

        ## token m1,2,3
        #('logdirs/20180201/karel-lgrl-ref-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #250100,
        #'1'),
        #('logdirs/20180201/karel-lgrl-ref-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #350100,
        #'0,1'),
        #('logdirs/20180201/karel-lgrl-ref-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #400100,
        #'0,0,1'),

        ## edit m1
        #('logdirs/20180207/karel-lgrl-ref-edit-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #  100100,
        #  '1'),
        #('logdirs/20180207/karel-lgrl-ref-edit-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #  350100,
        #  '0,1'),
        #('logdirs/20180207/karel-lgrl-ref-edit-m1-sgd-cl1-lr0.1-lds100k-ldr0.5',
        # 100100,
        #  '0,0,1'),

        # edit m2
        ('logdirs/20180211/karel-lgrl-ref-edit-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
          436300,
          '1'),
        ('logdirs/20180211/karel-lgrl-ref-edit-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
          400100,
          '0,1'),
        ('logdirs/20180211/karel-lgrl-ref-edit-m12-sgd-cl1-lr0.1-lds100k-ldr0.5',
         300100,
          '0,0,1'),

        ## edit m1,2,3
        #('logdirs/20180207/karel-lgrl-ref-edit-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #400100,
        #'1'),
        #('logdirs/20180207/karel-lgrl-ref-edit-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #436300,
        #'0,1'),
        #('logdirs/20180207/karel-lgrl-ref-edit-m123-sgd-cl1-lr0.1-lds100k-ldr0.5',
        #400100,
        #'0,0,1'),
        ]:
    print('python eval.py --model_type karel-lgrl-ref '
          '--dataset karel --max_beam_trees 64 --step {step} '
          '--karel-mutate-ref --karel-mutate-n-dist {dist} '
          '--model_dir {logdir} '
          '--eval-final '
          '--report-path {logdir}/report-test-m{dist}-{step}.jsonl '
          '--hide-example-info').format(
              step=ckpt_number, logdir=logdir, dist=dist)
