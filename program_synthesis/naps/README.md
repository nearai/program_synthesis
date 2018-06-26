# NAPS -- Natural Program Synthesis Dataset
The structure of the NAPS dataset is thoroughly described in the paper: [To be added later]

## Download the dataset
Use the link to download the dataset: https://goo.gl/WaBdbb

## Examples

Solution description: `given strings var0, var1 . if length of var0 not equals to the length of var1 , return "NO". if both var0 and var1 contain at least one character '1' or both of them do not contain '1' at all , return "YES"; else return "NO" .`

Pretty print of UAST: 
```
func char** __main__(char* var0, char* var1)
  vars:  int* var2,  int var3,  int var4,  int var5,  char** var6
  var6 = new char**()
  var2 = new int*(len(var0))
  var3 = 0
  var4 = 0
  if (len(var0) != len(var1))
    var6 = array_concat(var6, string_split("NO", " \t"))
    return var6
  else
    pass
  var5 = 0
  for(; (var5 < len(var0)); var5 = (var5 + 1))
    if (var0[var5] == 49)
      var3 = 1
    else
      pass
    if (var1[var5] == 49)
      var4 = 1
    else
      pass

  if (var3 == var4)
    var6 = array_concat(var6, string_split("YES", " \t"))
  else
    var6 = array_concat(var6, string_split("NO", " \t"))
  return var6
```

## Working with the dataset
[pipelines/](pipelines/) provides examples on how to work with the dataset.
* [pipelines/read_naps.py](pipelines/read_naps.py) demonstrates how to use [pipes/](pipes/) to read the dataset with batching, shuffling, etc;
* [pipelines/validate_naps.py](pipelines/validate_naps.py) uses the execution engine from [uast/uast.py](uast/uast.py) to validate that all solutions in the dataset pass the tests.
Note, though since solutions are human-written some of them might have unintended minor non-determinism caused by e.g. floating point arithmetic, which might result in one or two stray solutions
failing to pass some of the tests;
* [pipelines/pring_naps.py](pipelines/print_naps.py) demonstrates how one might use [uast/uast_pprint.py](uast/uast_pprint.py) to print UAST in nice human-readable format.

## UAST specification
See [uast/README.md](uast/README.md) for the formal specification of UAST.  

## Baseline models
[To be added later]
