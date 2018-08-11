import json

import pandas as pd
import torch
from tqdm import tqdm

import program_synthesis.karel.dataset.dataset as dataset
import program_synthesis.karel.dataset.refine_env as env
import program_synthesis.karel.dataset.utils as utils


def random_search(example, debug=True):
    kenv = env.KarelRefineEnv(example.input_tests)

    if debug:
        print("Target code:")
        print(utils.beautify(' '.join(example.code_sequence)))

    for it in tqdm(range(250), disable=True):
        kenv.reset()

        for ts in range(20):
            act = kenv.action_space.sample()
            obs, reward, done, _ = kenv.step(act)

            if done:

                if debug:
                    print(f"Found on iteration {it}")
                    kenv.render()
                return True

    if debug:
        print("Fail!")
    return False


def refactor():
    with open('random_search.json') as f:
        stats = json.load(f)

    info = {
        'solved': [],
        'length': []
    }

    test_ds = dataset.KarelTorchDataset(dataset.relpath('../../data/karel/{}{}.pkl'.format('test', '')), lambda x: x)
    dataset_loader = torch.utils.data.DataLoader(test_ds, collate_fn=lambda x: x)

    for (example,) in dataset_loader:
        idx = example.idx
        info['solved'].append(stats[str(idx)])
        info['length'].append(len(example.code_sequence))

    df = pd.DataFrame(info)
    print(df)

    with open('random_search.csv', 'w') as f:
        f.write(df.to_csv())


def main(debug):
    test_ds = dataset.KarelTorchDataset(dataset.relpath('../../data/karel/{}{}.pkl'.format('test', '')), lambda x: x)
    dataset_loader = torch.utils.data.DataLoader(test_ds, collate_fn=lambda x: x)

    good = 0
    total = 0

    with open('random_search.json') as f:
        stats = json.load(f)

    for (example,) in dataset_loader:
        if str(example.idx) in stats:
            good += int(stats[str(example.idx)])
            total += 1
            continue

        print("Example:", example.idx)

        ok = random_search(example, debug=debug)

        good += int(ok)
        total += 1

        stats[example.idx] = ok

        with open('random_search.json', 'w') as f:
            json.dump(stats, f)

        print(f"\n{good}/{total} = {good/total * 100:.3}%\n")


if __name__ == '__main__':
    main(True)
    # refactor()
