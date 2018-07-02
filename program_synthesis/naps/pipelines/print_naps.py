"""
Print several solutions from the NAPS dataset.
"""
from program_synthesis.naps.pipelines.read_naps import read_naps_dataset

from program_synthesis.naps.uast import uast_pprint


if __name__ == "__main__":
    for name, ds in zip(("trainA", "trainB", "test"), read_naps_dataset()):
        print("DATASET %s" % name)
        with ds:
            for d, _ in zip(ds, range(5)):
                if "is_partial" in d and d["is_partial"]:
                    continue
                print(' '.join(d["text"]))
                uast_pprint.pprint(d["code_tree"])
                print()
