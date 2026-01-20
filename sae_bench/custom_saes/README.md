There are a few requirements for the SAE object. If your SAE object inherits `BaseSAE`, then most of these will be inherited from the `BaseSAE`.

- It must have `encode()`, `decode()`, and `forward()` methods.
- The evals of SCR, TPP, and feature absorption require a `W_dec`, which is an nn.Parameter initialized with the following shape: `self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))`.
- `W_dec` should have unit norm decoder vectors. Some SAE trainers do not enforce this. `BaseSAE` has a function `check_decoder_norms()`, which we recommend calling when loading the SAE. For an example of how to fix this, refer to `normalize_decoder()` in `relu_sae.py`.
- The SAE must have a `dtype` and `device` attribute.
- The SAE must have a `.cfg` field, which contains attributes like `d_sae` and `d_in`. The core evals utilize SAE Lens internals, and require a handful of blank fields, which are already set in the `CustomSaeConfig` dataclass.
- In general, we recommend modifying an existing SAE class, such as the `relu_sae.py` class. You will have to modify `encode()` and `decode()`, and will probably have to add a function to load your state dict.

Refer to `SAEBench/sae_bench_demo.ipynb` for an example of how to compare a custom SAE with a baseline SAE and create some graphs. There is also a cell demonstrating how to run all evals on a selection of SAEs.

If your SAEs are trained with the [dictionary_learning repo](https://github.com/saprmarks/dictionary_learning), you can evaluate your SAEs on all evaluations by setting the HuggingFace repo name and model name. Refer to the docstring in `SAEBench/custom_saes/run_all_evals_dictionary_learning_saes.py` to learn more.

If you want a python script to evaluate your custom SAEs, refer to `run_all_evals_custom_saes.py`. You will have to handle loading the custom SAEs. There's an example provided.

If there are any pain points when using this repo with custom SAEs, please do not hesitate to reach out or raise an issue.

# Evaluation Options

If you want a more granular approach to running evaluations, we provide two options.

1. **Using Evaluation Templates**:

   - Use the secondary `if __name__ == "__main__"` block in each `main.py`
   - Results are saved in SAE Bench format for easy visualization
   - Compatible with provided plotting tools

2. **Direct Function Calls**:
   - Use `run_eval_single_sae()` in each `main.py`
   - Simpler interface requiring only model, SAE, and config values
   - Graphing will require manual formatting
