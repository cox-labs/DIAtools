# MLprediction

Preprocessing and postprocessing scripts to handle data to/from MSMS spectrum predictors (DeepMass/WiNNer/Prosit) in order
to produce _in silico_ predicted library for MaxDIA.

DeepMass/WiNNer
https://doi.org/10.1038/s41592-019-0427-6

Prosit
https://doi.org/10.1038/s41592-019-0426-7

## Preprocess
It digests protein records from fasta files using specified parameters and generates an input formatted file for MSMS predictors
```
python preprocess/main.py deepmass preprocess/parameters.json
python preprocess/main.py winner preprocess/parameters.json
python preprocess/main.py prosit preprocess/parameters.json
```

### Dependencies
| Package    | Version |
|:---------- |:------- |
| argparse   | 1.1     |
| json       | 2.0.9   |
| re         | 2.2.1   |

## Postprocess
Taking predicted MSMS spectrum, this scripts generates evidence/msms/peptide files, that are necessary to run MaxDIA.
Additionally to the data integration, it allows to predict Retention Time using training dataset.
```
python postprocess/main.py deepmass postprocess/parameters.json
python postprocess/main.py winner postprocess/parameters.json
python postprocess/main.py prosit postprocess/parameters.json
```


### Dependencies
| Package    | Version |
|:---------- |:------- |

