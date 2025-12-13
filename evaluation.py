import os
import asyncio
import numpy as np
import torch
import wandb
import argparse
import json
from pathlib import Path
from settings import settings
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from data import create_dataloaders
from models import SAETransformer, SAETransformerOutput
from utils.io import (
    load_config, 
    save_activation_data_to_wandb,
    save_metrics_to_wandb,
    save_explanations_to_wandb,
    load_activation_data_from_wandb,
    load_metrics_from_wandb,
    load_explanations_from_wandb
)
from utils.metrics import explained_variance, get_activations_for_sae_type, compute_alive_dictionary_indices
from utils.plotting import create_pareto_plots
from auto_interp.explainers.features import FeatureRecord, Feature
from auto_interp.explainers.explainer import DefaultExplainer, ExplainerResult
from auto_interp.clients import OpenAIClient, TogetherAIClient
from auto_interp.explainers.sampler import stratified_sample_by_max_activation
from auto_interp.scorers.classifier.detection import DetectionScorer
from auto_interp.scorers.classifier.fuzz import FuzzingScorer


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run SAE evaluation including activation data collection, neuron explanation generation, and analysis.
    """
    
    # Set Wandb cache directories to use output_path instead of home directory
    output_path_abs = Path(args.output_path).absolute()
    output_path_abs.mkdir(parents=True, exist_ok=True)
    
    wandb_cache_dir = output_path_abs / "wandb_cache"
    wandb_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables BEFORE any Wandb operations
    os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DATA_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DIR"] = str(wandb_cache_dir)
    os.environ["TMPDIR"] = str(output_path_abs)  # Also set general temp directory
    
    print(f"Using Wandb cache directory: {wandb_cache_dir}")
    print(f"Using temp directory: {output_path_abs}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    runs = api.runs(args.wandb_project)

    if args.filter_runs_by_name is not None:
        old_len = len(runs)
        runs = [run for run in runs if args.filter_runs_by_name in run.name]
        print(f"Found {len(runs)}/{old_len} runs matching filter: {args.filter_runs_by_name}")

    # Collect all metrics for pareto plots
    all_run_metrics = []

    for run in runs:
        run_id = run.id
        run_config = run.config
        run_config['data']['n_eval_samples'] = args.n_eval_samples
        try:
            model = SAETransformer.from_wandb(f"{args.wandb_project}/{run_id}").to(device)
            model.saes.eval()
            if args.sae_position is None:
                raw_sae_positions = model.raw_sae_positions
            else:
                assert args.sae_position in model.raw_sae_positions, f"SAE position {args.sae_position} not found in model"
                raw_sae_positions = [args.sae_position]
        except Exception as e:
            print(f"Error loading model from Wandb: {e}")
            continue
        
        # Override n_train_samples if specified (to avoid slow data skipping)
        if args.override_n_train_samples is not None:
            print(f"Overriding n_train_samples: {run_config['data']['n_train_samples']} -> {args.override_n_train_samples}")
            run_config['data']['n_train_samples'] = args.override_n_train_samples
            
        config: Config = load_config(run_config, Config)

        # Initialize Wandb run for this specific run (resume original run)
        wandb.init(
            project=args.wandb_project.split("/")[-1],  # Extract project name
            entity=args.wandb_project.split("/")[0],    # Extract entity  
            id=run_id,  # Use the same run ID
            resume="allow",  # Allow resuming existing run
            reinit="finish_previous",
            dir=str(wandb_cache_dir)  # Explicitly set Wandb's working directory
        )

        metrics = {}
        accumulated_data = None
        all_token_ids = None
        loaded_metrics = None
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)

        # Try to load existing data from Wandb
        print(f"Attempting to load activation data from Wandb for run {run_id}...")
        try:
            accumulated_data, all_token_ids = load_activation_data_from_wandb(
                run_id, project=args.wandb_project, output_path=args.output_path
            )
            
            print(f"Successfully loaded activation data from Wandb run files")
            if all_token_ids is not None:
                print(f"Loaded token IDs from Wandb run files")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No existing activation data found: {e}")
            print("Will compute activation data and metrics from scratch")
        
        # Try to load existing metrics separately
        try:
            loaded_metrics = load_metrics_from_wandb(run_id, project=args.wandb_project, output_path=args.output_path)
            if loaded_metrics is not None and not args.force_recompute:
                metrics = loaded_metrics
                print(f"Loaded existing metrics from Wandb for {len(metrics)} SAE positions")
            elif loaded_metrics is not None and args.force_recompute:
                print(f"Found existing metrics but --force_recompute is set, will recompute")
        except Exception as e:
            print(f"No existing metrics found: {e}")
            print("Will compute metrics from scratch")

        # Try to load existing explanations if needed
        all_explanation_scores = None
        if args.generate_explanations:
            try:
                loaded_explanations = load_explanations_from_wandb(run_id, project=args.wandb_project, output_path=args.output_path)
                if loaded_explanations is not None and not args.force_recompute and not args.force_recompute_explanations:
                    all_explanation_scores = loaded_explanations
                    print(f"Loaded existing explanations from Wandb")
                elif loaded_explanations is not None and (args.force_recompute or args.force_recompute_explanations):
                    print(f"Found existing explanations but force recompute is set, will recompute")
            except Exception as e:
                print(f"No existing explanations found: {e}")
                print("Will compute explanations from scratch")
        if accumulated_data is None or len(metrics) == 0:
            print(f"Obtaining features for {run_id}")

            # Load model and dataloader
            _, eval_loader = create_dataloaders(data_config=config.data, global_seed=config.seed, quick_eval=True)
            total_tokens = 0
            all_token_ids: list[list[str]] = []

            # Create placeholder tensors for efficient batch accumulation
            # Note: For feature extraction, 'nonzero_activations' stores probabilities for Bayesian SAEs, activations for ReLU SAEs
            accumulated_data = {}
            for sae_pos in raw_sae_positions:
                accumulated_data[sae_pos] = {
                    'nonzero_activations': [],
                    'data_indices': [],
                    'neuron_indices': [],
                }
                metrics[sae_pos] = {
                    'alive_dict_components': set(),
                    'sparsity_l0': 0.0,
                    'mse': 0.0,
                    'explained_variance': 0.0,
                }

            total_tokens = 0
            for batch in tqdm(eval_loader, desc="Processing batches"):
                token_ids = batch[config.data.column_name].to(device)
                
                # Reshape token_ids to break into chunks of window_size
                batch_size, seq_len = token_ids.shape
                if seq_len % args.window_size != 0:
                    raise ValueError(f"Sequence length {seq_len} is not divisible by window_size {args.window_size}")

                num_chunks = seq_len // args.window_size
                chunked_batch_size = batch_size * num_chunks
                token_ids_chunked = token_ids.view(batch_size, num_chunks, args.window_size)
                token_ids_chunked = token_ids_chunked.reshape(chunked_batch_size, args.window_size)
                
                n_tokens = token_ids_chunked.shape[0] * token_ids_chunked.shape[1]
                total_tokens += n_tokens

                # Run through the SAE-augmented model
                with torch.no_grad():
                    output: SAETransformerOutput = model.forward(
                        tokens=token_ids_chunked,
                        sae_positions=model.raw_sae_positions,
                        compute_loss=True,
                    )

                for sae_pos in raw_sae_positions:
                    sae_output = output.sae_outputs[sae_pos]
                    
                    # Compute MSE using the same logic as utils/metrics.py
                    mse_val = mse_loss(
                        sae_output.output,
                        sae_output.input,
                        reduction='mean'
                    ).item()
                    metrics[sae_pos]['mse'] += mse_val * n_tokens
                    
                    # Compute explained variance using the shared function from utils.metrics
                    exp_var = explained_variance(
                        sae_output.output,
                        sae_output.input,
                        layer_norm_flag=False
                    ).mean().item()
                    metrics[sae_pos]['explained_variance'] += exp_var * n_tokens
                    
                    # Get activations using the shared function from utils.metrics
                    acts = get_activations_for_sae_type(sae_output, config.saes.sae_type)

                    # Compute L0 sparsity using the same logic as utils/metrics.py
                    l0_val = torch.norm(acts, p=0, dim=-1).mean().item()
                    metrics[sae_pos]['sparsity_l0'] += l0_val * n_tokens
                    
                    # Compute alive dictionary components using the shared helper function
                    alive_indices = compute_alive_dictionary_indices(acts)
                    metrics[sae_pos]['alive_dict_components'].update(alive_indices)

                    if args.save_activation_data:
                        # Collect non-zero activations for explanation generation
                        data_indices, neuron_indices = acts.sum(1).nonzero(as_tuple=True)
                        if data_indices.numel() > 0:
                            # Extract all relevant activations at once (N, seq_len)
                            nonzero_activations = acts[data_indices, :, neuron_indices]

                            # Add the offset to the data indices for global indexing
                            global_data_indices = data_indices + len(all_token_ids)

                            # Accumulate tensors for this SAE position
                            accumulated_data[sae_pos]['nonzero_activations'].append(nonzero_activations.to(torch.float16).cpu())
                            accumulated_data[sae_pos]['data_indices'].append(global_data_indices.cpu())
                            accumulated_data[sae_pos]['neuron_indices'].append(neuron_indices.cpu())

                # Store tokenized sequences for explanation generation
                chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
                all_token_ids.extend(chunked_tokens)

            for sae_pos in raw_sae_positions:
                if args.save_activation_data:
                    accumulated_data[sae_pos]['nonzero_activations'] = torch.cat(accumulated_data[sae_pos]['nonzero_activations'], dim=0).contiguous()
                    accumulated_data[sae_pos]['data_indices'] = torch.cat(accumulated_data[sae_pos]['data_indices'], dim=0).contiguous()
                    accumulated_data[sae_pos]['neuron_indices'] = torch.cat(accumulated_data[sae_pos]['neuron_indices'], dim=0).contiguous()

                metrics[sae_pos]['sparsity_l0'] /= total_tokens
                metrics[sae_pos]['mse'] /= total_tokens
                metrics[sae_pos]['explained_variance'] /= total_tokens
                
                # Convert alive components set to count and proportion
                alive_components = metrics[sae_pos]['alive_dict_components']
                num_alive = len(alive_components)
                total_dict_size = model.saes[sae_pos.replace(".", "-")].n_dict_components
                metrics[sae_pos]['alive_dict_components'] = num_alive
                metrics[sae_pos]['alive_dict_components_proportion'] = num_alive / total_dict_size
            
            # Always save metrics
            print("Saving metrics to Wandb...")
            try:
                save_metrics_to_wandb(metrics=metrics, output_path=args.output_path)
            except Exception as e:
                print(f"Warning: Failed to upload metrics to Wandb: {e}")

            # Save activation data to Wandb
            if args.save_activation_data:
                print("Saving accumulated activation data to Wandb...")
                save_activation_data_to_wandb(
                    accumulated_data=accumulated_data,
                    all_token_ids=all_token_ids,
                    output_path=args.output_path,
                    skip_upload=args.skip_upload,
                    chunk_upload=not args.no_chunk_upload  # Default is True unless disabled
                )

        # Collect metrics for pareto plot
        run_metrics = {
            'run_id': run_id,
            'run_name': run.name,
            'config': config,
            'metrics': metrics
        }
        
        all_run_metrics.append(run_metrics)
        
        if args.generate_explanations and all_explanation_scores is None:
            if accumulated_data is None:
                print("No activation data found, skipping explanation generation")
                continue
            
            # Initialize dict to store all explanation scores for this run
            all_explanation_scores = {}

            # Initialize explainer
            explainer = DefaultExplainer(
                client=OpenAIClient(
                    api_key=settings.openai_api_key,
                    model=args.explanation_model,
                ),
                tokenizer=tokenizer,
                cot=True,
                threshold=0.3,
                activations=False,
                temperature=0.0,
            )

            for sae_pos in raw_sae_positions:
                data = accumulated_data[sae_pos]
                all_explanation_scores[sae_pos] = []
                
                # Count occurrences of each neuron and calculate total activation
                unique_neurons = torch.unique(data['neuron_indices'], return_counts=False)
                
                # Efficient computation using scatter_reduce
                num_neurons = unique_neurons.max().item() + 1
                seq_len = data['nonzero_activations'].shape[1]
                
                # Initialize tensor to store max activations per neuron per position
                # neuron_total_activations_full = torch.zeros(num_neurons, seq_len, 
                #                                            device=data['nonzero_activations'].device,
                #                                            dtype=torch.float32)
                
                # Use scatter_reduce to get max activation per neuron per position
                # neuron_total_activations_full.scatter_reduce_(
                #     0,  # dimension to scatter along
                #     data['neuron_indices'].unsqueeze(1).expand(-1, seq_len),  # indices
                #     data['nonzero_activations'].float(),  # values to scatter
                #     reduce='amax',  # take maximum
                #     include_self=False
                # )
                
                # Extract only the neurons that actually appear in the data
                # neuron_total_activations = neuron_total_activations_full[unique_neurons]
                
                # Count number of unique sequences each neuron activates on
                # More efficient vectorized approach
                    
                # Create unique (neuron, sequence) pairs to avoid double counting
                neuron_sequence_pairs = torch.stack([data['neuron_indices'], data['data_indices']], dim=1)
                unique_pairs = torch.unique(neuron_sequence_pairs, dim=0)
                
                # Count sequences per neuron using bincount
                neuron_indices_from_pairs = unique_pairs[:, 0]
                neuron_sequence_counts = torch.bincount(neuron_indices_from_pairs, minlength=num_neurons)
                
                # Get counts only for neurons that appear
                counts_for_active_neurons = neuron_sequence_counts[unique_neurons]
                
                min_required_examples = args.num_features_to_explain + args.num_positive_examples
                sufficient_examples_mask = counts_for_active_neurons >= min_required_examples
                filtered_indices = torch.arange(len(unique_neurons))[sufficient_examples_mask]
                
                if len(filtered_indices) == 0:
                    print(f"  No neurons found with at least {min_required_examples} activated sequences")
                    continue
                
                num_to_sample = min(args.num_neurons, len(filtered_indices))
                torch.manual_seed(config.seed)
                random_perm = torch.randperm(len(filtered_indices))
                sampled_indices = filtered_indices[random_perm[:num_to_sample]]
                sampled_neurons = unique_neurons[sampled_indices]
                print(f"SAE position {sae_pos}: {len(unique_neurons)} total neurons, {len(filtered_indices)} with ≥{min_required_examples} examples, randomly sampling {len(sampled_neurons)}")
                print(f"  Processing {len(sampled_neurons)} neurons for explanation...")
                
                # Initialize running statistics
                detection_scores = []
                fuzz_scores = []
                
                # Process each neuron for explanation
                pbar = tqdm(sampled_neurons, desc="Processing neurons")
                for i, neuron_idx in enumerate(pbar):
                    neuron_idx_item = neuron_idx.item()
                    
                    # Update progress bar with running statistics
                    stats_str = ""
                    if detection_scores:
                        stats_str += f"det_mean={np.mean(detection_scores):.3f}"
                    if fuzz_scores:
                        if stats_str:
                            stats_str += ", "
                        stats_str += f"fuzz_mean={np.mean(fuzz_scores):.3f}"
                    if stats_str:
                        pbar.set_postfix_str(stats_str)
                    
                    try:
                        feature = Feature(sae_pos=sae_pos, neuron_idx=neuron_idx_item)
                        # Compute minimum required examples as explanation + positive examples
                        min_required_examples = args.num_features_to_explain + args.num_positive_examples
                        feature_record = FeatureRecord.from_data(
                            data=data,
                            feature=feature,
                            all_token_ids=all_token_ids,
                            neuron_idx=neuron_idx,
                            num_explanation_examples=args.num_features_to_explain,
                            num_positive_examples=args.num_positive_examples,
                            num_negative_examples=args.num_negative_examples,
                            stratified_quantiles=args.stratified_quantiles,
                            min_examples_required=min_required_examples,
                            seed=config.seed,
                        )
                        
                        # Skip if we couldn't create a valid feature record
                        if feature_record is None:
                            print(f"  Skipping neuron {neuron_idx_item} - not enough examples")
                            continue

                        # Generate explanation using the explanation_examples
                        explanation: ExplainerResult = asyncio.run(explainer(feature_record))

                        # Score the explanation if requested
                        print(f"  Neuron {neuron_idx_item}: {explanation.explanation}")
                        
                        # Create scoring client for Detection and Fuzz scorers
                        score_client = TogetherAIClient(
                            api_key=settings.together_ai_api_key,  # Use together API key
                            model=args.scoring_model
                        )
                        
                        # 1. Detection Score
                        print(f"    Computing Detection score...")
                        detection_scorer = DetectionScorer(
                            client=score_client,
                            verbose=False,
                            batch_size=5,
                            use_structured_output=True,
                            temperature=0.0,
                        )
                        detection_result = asyncio.run(detection_scorer(explanation))
                        detection_score = detection_result.score
                        print(f"    ✓ Detection score: {detection_score:.3f}")
                        detection_scores.append(detection_score)
                        
                        # 2. Fuzz Score
                        # Commented out for now - taking too long
                        print(f"    Computing Fuzz score...")
                        fuzz_scorer = FuzzingScorer(
                            client=score_client,
                            verbose=False,
                            batch_size=5,
                            threshold=0.3,
                            use_structured_output=True,
                            temperature=0.0,
                        )
                        fuzz_result = asyncio.run(fuzz_scorer(explanation))
                        # The score is now directly the accuracy
                        fuzz_score = fuzz_result.score
                        print(f"    ✓ Fuzz score: {fuzz_score:.3f}")
                        fuzz_scores.append(fuzz_score)
                        
                        # Store scores
                        all_explanation_scores[sae_pos].append({
                            'neuron_idx': neuron_idx_item,
                            'explanation': explanation.explanation,
                            'detection_score': detection_score,
                            'fuzz_score': fuzz_score,
                        })
                        
                    except Exception as e:
                        print(f"  ✗ Error processing neuron {neuron_idx_item}: {e}")
                        print(f"    Skipping this neuron and continuing...")
                        continue
                
                # Print final statistics for this SAE position
                print(f"\n  Final statistics for {sae_pos}:")
                if detection_scores:
                    print(f"    Detection: mean={np.mean(detection_scores):.3f}, std={np.std(detection_scores):.3f}, n={len(detection_scores)}")
                if fuzz_scores:
                    print(f"    Fuzz:      mean={np.mean(fuzz_scores):.3f}, std={np.std(fuzz_scores):.3f}, n={len(fuzz_scores)}")
            
            # Save explanations to Wandb
            try:
                print(f"Saving explanations to Wandb for run {run_id}")
                save_explanations_to_wandb(explanations=all_explanation_scores, output_path=args.output_path)
                print(f"Successfully uploaded explanations to Wandb for run {run_id}")
                
            except Exception as e:
                print(f"Warning: Failed to upload explanations to Wandb: {e}")
    
        # Display explanation summary for this run if explanations were generated or loaded
        if args.generate_explanations and all_explanation_scores:
            print("\n" + "=" * 60)
            print(f"EXPLANATION SCORES SUMMARY - Run {run_id}")
            print("=" * 60)
            
            # Save scores to JSON for this run
            scores_path = Path(args.output_path) / f"explanation_scores_{run_id}.json"
            with open(scores_path, 'w') as f:
                json.dump(all_explanation_scores, f, indent=2)
            print(f"\nScores saved to: {scores_path}")
            
            # Display summary statistics by layer
            print("\nScore Statistics by Layer:")
            all_scores = []  # For overall top explanations
            
            for sae_pos, scores_list in all_explanation_scores.items():
                if not scores_list:
                    continue
                    
                # Get scores for this layer
                layer_detection_scores = [s['detection_score'] for s in scores_list if s['detection_score'] is not None]
                layer_fuzz_scores = [s['fuzz_score'] for s in scores_list if s['fuzz_score'] is not None]
                
                print(f"\n  {sae_pos}:")
                print(f"    Total neurons: {len(scores_list)}")
                if layer_detection_scores:
                    print(f"    Detection: mean={np.mean(layer_detection_scores):.3f}, std={np.std(layer_detection_scores):.3f}, n={len(layer_detection_scores)}")
                if layer_fuzz_scores:
                    print(f"    Fuzz:      mean={np.mean(layer_fuzz_scores):.3f}, std={np.std(layer_fuzz_scores):.3f}, n={len(layer_fuzz_scores)}")
                
                # Add to overall list with sae_pos for top explanations display
                for score_dict in scores_list:
                    score_dict_with_pos = score_dict.copy()
                    score_dict_with_pos['sae_pos'] = sae_pos
                    all_scores.append(score_dict_with_pos)

            # Display overall statistics
            overall_detection_scores = [s['detection_score'] for s in all_scores if s['detection_score'] is not None]
            overall_fuzz_scores = [s['fuzz_score'] for s in all_scores if s['fuzz_score'] is not None]
            
            print(f"\n  Overall (All Layers):")
            print(f"    Total neurons: {len(all_scores)}")
            if overall_detection_scores:
                print(f"    Detection: mean={np.mean(overall_detection_scores):.3f}, std={np.std(overall_detection_scores):.3f}, n={len(overall_detection_scores)}")
            if overall_fuzz_scores:
                print(f"    Fuzz:      mean={np.mean(overall_fuzz_scores):.3f}, std={np.std(overall_fuzz_scores):.3f}, n={len(overall_fuzz_scores)}")

            # Display top scoring explanations across all layers
            print("\nTop 5 explanations by Detection score (across all layers):")
            sorted_by_detection = sorted([s for s in all_scores if s['detection_score'] is not None], 
                                        key=lambda x: x['detection_score'], reverse=True)[:5]
            for i, score_entry in enumerate(sorted_by_detection, 1):
                print(f"  {i}. Neuron {score_entry['neuron_idx']} ({score_entry['sae_pos']}): {score_entry['detection_score']:.3f}")
                print(f"     {score_entry['explanation'][:100]}...")
    
        # Finish the current wandb run before moving to the next one
        wandb.finish()

    wandb.teardown()
    # Create pareto plots after processing all runs
    create_pareto_plots(all_run_metrics)


def main():
    """Main function with argparse configuration."""
    parser = argparse.ArgumentParser(description="Run SAE evaluation with configurable parameters")
    
    # Window and processing parameters
    parser.add_argument("--window_size", type=int, default=64, 
                       help="Size of token windows for processing (default: 64)")

    # Explanation parameters
    parser.add_argument("--num_neurons", type=int, default=300,
                       help="Number of top neurons to process per layer (default: 300)")
    parser.add_argument("--num_positive_examples", type=int, default=100,
                       help="Number of positive examples (where neuron is active) for scoring (default: 100)")
    parser.add_argument("--num_negative_examples", type=int, default=100,
                       help="Number of negative examples (where neuron is inactive) for scoring (default: 100)")
    parser.add_argument("--max_activated_features_per_neuron", type=int, default=10000,
                       help="Maximum number of activated features to use for explanation per neuron (default: 1000)")
    parser.add_argument("--num_features_to_explain", type=int, default=10,
                       help="Number of top activation examples to use for explanation (default: 10)")
    parser.add_argument("--n_eval_samples", type=int, default=10000,
                       help="Number of evaluation samples to process (default: 50000)")
    parser.add_argument("--stratified_quantiles", type=int, default=20,
                       help="Number of quantiles for stratified sampling of activation examples and neurons (default: None)")
    
    # Model parameters
    parser.add_argument("--explanation_model", type=str, default="gpt-4o",
                       help="Model to use for generating explanations (default: gpt-4o)")
    parser.add_argument("--scoring_model", type=str, default="llama-3.3-70b",
                       help="Model to use for scoring explanations (default: llama-3.3-70b)")
    
    # Wandb and storage parameters
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project' (default: raymondl/tinystories-1m)")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by a specific string in their name (default: None)")
    parser.add_argument("--output_path", type=str, default="./artifacts",
                       help="Path for storing temporary files and artifacts (default: ./artifacts)")
    
    # Execution flags
    parser.add_argument("--save_activation_data", action="store_true", default=False,
                       help="Save activation data (default: False)")
    
    parser.add_argument("--skip_upload", action="store_true", default=False,
                       help="Skip uploading to Wandb, only save locally (default: False)")
    
    parser.add_argument("--no_chunk_upload", action="store_true", default=False,
                       help="Disable chunked upload (upload all files at once) (default: False)")
    
    parser.add_argument("--generate_explanations", action="store_true", default=False,
                       help="Generate neuron explanations (default: False)")

    parser.add_argument("--force_recompute", action="store_true", default=False,
                       help="Force recomputation of metrics even if existing ones are found (default: False)")

    parser.add_argument("--force_recompute_explanations", action="store_true", default=False,
                       help="Force recomputation of explanations only, keeping existing metrics and activations (default: False)")

    # For debugging
    parser.add_argument("--override_n_train_samples", type=int, default=None,
                       help="Override n_train_samples to avoid slow data skipping (default: None - use config value)")
    
    parser.add_argument("--sae_position", type=str, default=None,
                       help="SAE position to evaluate (default: None - use all)")

    args = parser.parse_args()
    
    # Run the evaluation with parsed arguments
    run_evaluation(args)


if __name__ == "__main__":
    main()
