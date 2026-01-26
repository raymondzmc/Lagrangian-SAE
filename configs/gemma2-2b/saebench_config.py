"""
Evaluation configuration file for SAEBench batch evaluation.

This file defines which runs to evaluate, organized by sparsity level (L0/K).
Each entry specifies the run name and optionally a checkpoint step.

Usage:
    python run_saebench.py --project gemma2-2b-65K --eval_config eval_config.py --eval_types core autointerp

    # Evaluate specific sparsity levels
    python run_saebench.py --project gemma2-2b-65K --eval_config eval_config.py --sparsity_levels 32 64

    # Evaluate specific methods
    python run_saebench.py --project gemma2-2b-65K --eval_config eval_config.py --methods lagrangian jumprelu
"""

# Dictionary mapping sparsity level -> method -> run config
# Each run config has:
#   - run_name: The wandb run display name (required)
#   - checkpoint_step: Optional step number to load specific checkpoint (if omitted, loads latest)
EVAL_RUNS = {
    32: {
        "relu": {
            "run_name": "relu_sparsity_coeff_35_n_dict_components_65536",
            # "checkpoint_step": 30000,  # Uncomment to use specific checkpoint
        },
        "gated": {
            "run_name": "gated_sparsity_coeff_0.025_n_dict_components_65536",
        },
        "topk": {
            "run_name": "topk_k_32_n_dict_components_65536",
        },
        "batch_topk": {
            "run_name": "batch_topk_k_32_n_dict_components_65536",
        },
        "jumprelu": {
            "run_name": "jumprelu_target_l0_32_n_dict_components_65536",
        },
        "matryoshka": {
            "run_name": "matryoshka_k_32_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha5": {
            "run_name": "matryoshka_lagrangian_target_l0_32_alpha_max_5_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha10": {
            "run_name": "matryoshka_lagrangian_target_l0_32_alpha_max_10_n_dict_components_65536",
        },
        "lagrangian_alpha5": {
            "run_name": "lagrangian_target_l0_32_alpha_max_5_n_dict_components_65536",
        },
        "lagrangian_alpha10": {
            "run_name": "lagrangian_target_l0_32_alpha_max_10_n_dict_components_65536",
        },
    },
    64: {
        "relu": {
            "run_name": "relu_sparsity_coeff_23_n_dict_components_65536",
        },
        "gated": {
            "run_name": "gated_sparsity_coeff_0.019_n_dict_components_65536",
        },
        "topk": {
            "run_name": "topk_k_64_n_dict_components_65536",
        },
        "batch_topk": {
            "run_name": "batch_topk_k_64_n_dict_components_65536",
        },
        "jumprelu": {
            "run_name": "jumprelu_target_l0_64_n_dict_components_65536",
        },
        "matryoshka": {
            "run_name": "matryoshka_k_64_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha5": {
            "run_name": "matryoshka_lagrangian_target_l0_64_alpha_max_5_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha10": {
            "run_name": "matryoshka_lagrangian_target_l0_64_alpha_max_10_n_dict_components_65536",
        },
        "lagrangian_alpha5": {
            "run_name": "lagrangian_target_l0_64_alpha_max_5_n_dict_components_65536",
        },
        "lagrangian_alpha10": {
            "run_name": "lagrangian_target_l0_64_alpha_max_10_n_dict_components_65536",
        },
    },
    128: {
        "relu": {
            "run_name": "relu_sparsity_coeff_14_n_dict_components_65536",
        },
        "gated": {
            "run_name": "gated_sparsity_coeff_0.0135_n_dict_components_65536",
        },
        "topk": {
            "run_name": "topk_k_128_n_dict_components_65536",
        },
        "batch_topk": {
            "run_name": "batch_topk_k_128_n_dict_components_65536",
        },
        "jumprelu": {
            "run_name": "jumprelu_target_l0_128_n_dict_components_65536",
        },
        "matryoshka": {
            "run_name": "matryoshka_k_128_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha5": {
            "run_name": "matryoshka_lagrangian_target_l0_128_alpha_max_5_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha10": {
            "run_name": "matryoshka_lagrangian_target_l0_128_alpha_max_10_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha5": {
            "run_name": "matryoshka_lagrangian_target_l0_128_alpha_max_5_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha10": {
            "run_name": "matryoshka_lagrangian_target_l0_128_alpha_max_10_n_dict_components_65536",
        },
        "lagrangian_alpha5": {
            "run_name": "lagrangian_target_l0_128_alpha_max_5_n_dict_components_65536",
        },
        "lagrangian_alpha10": {
            "run_name": "lagrangian_target_l0_128_alpha_max_10_n_dict_components_65536",
        },
    },
    256: {
        "relu": {
            "run_name": "relu_sparsity_coeff_8_n_dict_components_65536",
        },
        "gated": {
            "run_name": "gated_sparsity_coeff_9e-03_n_dict_components_65536",
        },
        "topk": {
            "run_name": "topk_k_256_n_dict_components_65536",
        },
        "batch_topk": {
            "run_name": "batch_topk_k_256_n_dict_components_65536",
        },
        "jumprelu": {
            "run_name": "jumprelu_target_l0_256_n_dict_components_65536",
        },
        "matryoshka": {
            "run_name": "matryoshka_k_256_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha5": {
            "run_name": "matryoshka_lagrangian_target_l0_256_alpha_max_5_n_dict_components_65536",
        },
        "matryoshka_lagrangian_alpha10": {
            "run_name": "matryoshka_lagrangian_target_l0_256_alpha_max_10_n_dict_components_65536",
        },
        "lagrangian_alpha5": {
            "run_name": "lagrangian_target_l0_256_alpha_max_5_n_dict_components_65536",
            "checkpoint_step": 20000,
        },
        "lagrangian_alpha10": {
            "run_name": "lagrangian_target_l0_256_alpha_max_10_n_dict_components_65536",
            "checkpoint_step": 20000,
        },
    },
}
