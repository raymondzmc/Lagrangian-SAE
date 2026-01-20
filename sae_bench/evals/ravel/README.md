## RAVEL Benchmark

#### Task 
RAVEL quantifies feature disentanglement. Given a dataset of entities (eg. cities) with multiple attributes (eg. country, language) we score an SAE's ability to have a precise causal effect on one of the attributes while leaving other attributes unaffected. The current form computes the disentanglement of two attributes `A` and `B` from a single file.

We can also train a Multi-task Distributed Alignment Search using the --train_mdas flag.


#### Implementation
The scoring consists of three steps:
1. Create a `RAVELInstance`, a dataset of Entity-Attribute pairs filterd to only contain pairs the model actually knows. The `RAVELInstance.prompts` contain tokenized prompts and more metadata. See `instance.py` and `generation.py`.
2. Select attribute-sprecific SAE latens with cosine similarity to MDBM (Multi Task Differentable Binary Masking).
3. Compute cause and isolation scores by intervening on attribute specific features
    - Cause evaluation: High accuracy if intervening with A_features is successful on base_A_template, ie. source_A_attribute_value is generated.
    - Isolation evaluation: High accuracy if intervening with B_features is unsuccessful on base_A_template, ie. base_A_attribute is generated regardless of intervention.
    - disentanglement_score is the mean: D = (cause_A[t] + cause_B[t] + isolation_AtoB[t] + isolation_BtoA[t]) / 4
    - see `intervention.py`

## Debugging Notes

- This eval suffers from memory fragmentation and intermittent OOMs when evaluating multiple SAEs. I fixed this on 3090s by decreasing the batch size from 32 -> 8. Moving to a larger GPU also works. It would be nice to improve this to enable larger batch sizes.