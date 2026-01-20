This repo implements k-sparse probing, where k can be any integer less than the SAE's hidden dim. The k-sparse probing is done on CPU using `sklearn`, and probe training speed can depend on CPU speed. By default, we evaluate k of [1, 2, 5], but this can be increased in `eval_config.py`. Probe training speed can slow significantly if using k > 10 and < 10 CPU cores.

Estimated runtime per dataset (currently there are 6 datasets):

- Pythia-70M: ~10 seconds to collect activations per layer with SAEs, ~20 seconds per SAE to perform probing
- Gemma-2-2B: ~2 minutes to collect activations per layer with SAEs, ~20 seconds per SAE to perform probing

Using Gemma-2-2B, at current batch sizes, I see a peak GPU memory usage of 22 GB. This fits on a 3090.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `evals/sparse_probing/` and run `python main.py`. It should produce `eval_results/sparse_probing/pythia-70m-deduped_layer_4_eval_results.json`.

`tests/test_sparse_probing.py` contains an end-to-end test of the sparse probing eval. Expected results are in `tests/test_data/sparse_probing_expected_results.json`. Running `pytest -s tests/test_sparse_probing` will verify that the actual results are within the specified tolerance of the expected results.

If the random seed is set, it's fully deterministic and results match perfectly using `compare_run_results.ipynb`. If the random seed is not set, results will vary up to 0.03 for some outlier values, with a mean difference of 0.005.