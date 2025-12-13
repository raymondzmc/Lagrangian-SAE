#!/usr/bin/env python3
"""
Device-aware SAE experiment sweep runner with separated config and sweep definitions.

Usage:
    python run_experiments.py --base_config configs/tinystories-gated.yaml --sweep_config sweep_configs/default_sweep.yaml
    python run_experiments.py --base_config configs/tinystories-hardconcrete.yaml --sweep_config sweep_configs/quick_test.yaml --devices cuda:0,cuda:1
"""
# Suppress Pydantic warnings about Field() attributes that don't apply in certain contexts
# (caused by jaxtyping's Annotated types used in Pydantic models)
import warnings
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field\\(\\).*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field\\(\\).*", category=UserWarning)

import os
import argparse
import subprocess
import time
import torch
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import logging
import copy
from itertools import product
from run import run

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_available_devices() -> List[str]:
    """Detect all available compute devices."""
    devices = []
    
    # Detect CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_id = f"cuda:{i}"
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                devices.append(device_id)
                logger.info(f"Found CUDA device: {device_id} ({memory_gb:.1f}GB) - {props.name}")
            except Exception as e:
                logger.warning(f"Error querying CUDA device {i}: {e}")
        
        # Add CPU as additional option when CUDA is available
        devices.append("cpu")
        logger.info("Found CPU device (additional option)")
        
    elif torch.backends.mps.is_available():
        # Use MPS only (don't add CPU when MPS is available)
        devices.append("mps")
        logger.info("Found MPS device (Apple Silicon)")
        
    else:
        # Fall back to CPU only
        devices.append("cpu")
        devices.append("cpu")
        logger.info("Using CPU device (no accelerators found)")
    
    return devices


def parse_device_list(device_str: str) -> List[str]:
    """Parse device specification string into list of device IDs."""
    if device_str.lower() == "auto":
        return detect_available_devices()
    else:
        return [d.strip() for d in device_str.split(',')]


def load_base_config(config_path: Path) -> Dict[str, Any]:
    """Load the base configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sweep_config(sweep_path: Path) -> Dict[str, Any]:
    """Load the sweep configuration file."""
    with open(sweep_path, 'r') as f:
        return yaml.safe_load(f)


def detect_sae_type(base_config: Dict[str, Any]) -> str:
    """Detect the SAE type from the base configuration."""
    sae_type = base_config.get('saes', {}).get('sae_type')
    if not sae_type:
        raise ValueError("SAE type not found in base config. Expected 'saes.sae_type' field.")
    return sae_type


def generate_parameter_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations based on sweep config, agnostic to parameter names."""
    
    # Reserved keys that are not parameters
    reserved_keys = {'max_experiments'}
    
    # Get all parameter grids from sweep config (list values)
    param_grids = {}
    # Get fixed parameters from sweep config (non-list values)
    fixed_params = {}
    
    for param_name, values in sweep_config.items():
        if param_name in reserved_keys:
            continue
        if isinstance(values, list):
            param_grids[param_name] = values
        else:
            # Non-list values are fixed parameters applied to all experiments
            fixed_params[param_name] = values
    
    if not param_grids:
        logger.warning("No parameter grids found in sweep config. Using single default combination.")
        return [fixed_params] if fixed_params else [{}]
    
    # Generate all combinations
    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())
    
    combinations = []
    for values in product(*param_values):
        combination = dict(zip(param_names, values))
        # Add fixed parameters to each combination
        combination.update(fixed_params)
        combinations.append(combination)
    
    # Apply max_experiments limit if specified
    max_experiments = sweep_config.get('max_experiments')
    if max_experiments and len(combinations) > max_experiments:
        logger.info(f"Limiting to {max_experiments} experiments (from {len(combinations)} total combinations)")
        combinations = combinations[:max_experiments]
    
    return combinations


def create_experiment_config(base_config: Dict[str, Any], params: Dict[str, Any], experiment_id: int) -> Dict[str, Any]:
    """Create a single experiment config by combining base config with parameter values."""
    
    # Deep copy the base config
    experiment_config = copy.deepcopy(base_config)
    
    # Define where different parameter types go
    top_level_params = {'lr', 'seed', 'gradient_accumulation_steps', 'max_grad_norm', 'lr_schedule', 
                       'min_lr_factor', 'warmup_samples', 'cooldown_samples', 'log_every_n_grad_steps',
                       'eval_every_n_samples', 'save_every_n_samples', 'wandb_project', 'wandb_run_name', 'wandb_tags'}
    
    data_params = {'n_train_samples', 'n_eval_samples', 'train_batch_size', 'eval_batch_size', 
                  'context_length', 'dataset_name', 'tokenizer_name'}

    # Apply parameter overrides
    for param_name, value in params.items():
        if param_name in top_level_params:
            # Top-level config parameters
            experiment_config[param_name] = value
        elif param_name in data_params:
            # Data section parameters
            data_section = experiment_config.setdefault('data', {})
            data_section[param_name] = value
        else:
            # Assume SAE parameters (most common case for sweeps)
            saes_section = experiment_config.setdefault('saes', {})
            saes_section[param_name] = value
    
    # Ensure required SAE parameters exist
    saes_section = experiment_config.setdefault('saes', {})
    # Create run name from parameters
    param_string_parts = []
    for param_name, value in params.items():
        if param_name in ['wandb_run_name', 'wandb_project', 'wandb_tags']:
            continue
        if isinstance(value, bool):
            param_string_parts.append(f"{param_name}_{str(value).lower()}")
        elif isinstance(value, float) and value < 0.01:
            param_string_parts.append(f"{param_name}_{value:.0e}")
        else:
            param_string_parts.append(f"{param_name}_{value}")
    param_string = "_".join(param_string_parts)
    run_name = experiment_config.get('wandb_run_name') or saes_section.get('sae_type')
    
    if param_string:
        experiment_config['wandb_run_name'] = f"{run_name}_{param_string}"
        experiment_config['wandb_tags'] = [str(run_name)] + [f"{k}_{v}" for k, v in params.items()]
    else:
        experiment_config['wandb_run_name'] = f"{run_name}_default"
        experiment_config['wandb_tags'] = [str(run_name)]
    
    # Add experiment metadata
    experiment_config['_experiment_id'] = f"exp_{experiment_id:03d}"
    experiment_config['_params'] = params
    
    return experiment_config


def run_single_experiment(config_path: Path, device_id: str, use_tmux: bool = True) -> tuple[str, int, str]:
    """Run a single experiment on the specified device."""
    start_time = time.time()
    config_name = config_path.name
    timestamp = time.strftime("%H%M%S")  # HHMMSS format
    experiment_name = f"{config_name.replace('.yaml', '')}_{device_id.replace(':', '_')}_{timestamp}"
    
    try:
        logger.info(f"[{device_id}] Starting experiment: {config_name}")
        
        if use_tmux:
            # Create tmux session for this experiment
            session_name = f"exp_{experiment_name}"
            
            # Kill existing session if it exists
            subprocess.run(["tmux", "kill-session", "-t", session_name], 
                         capture_output=True, check=False)
            
            # Create new tmux session and run experiment
            tmux_cmd = [
                "tmux", "new-session", "-d", "-s", session_name, 
                "-c", str(Path.cwd()),
                "python", "run.py", "--config_path", str(config_path), "--device", device_id
            ]
            
            logger.info(f"[{device_id}] Created tmux session: {session_name}")
            logger.info(f"[{device_id}] Monitor with: tmux attach -t {session_name}")
            
            # Start the tmux session
            result = subprocess.run(tmux_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"[{device_id}] Failed to create tmux session: {result.stderr}")
                return config_name, -4, f"Tmux session creation failed on {device_id}"
            
            # Wait for the session to complete and get exit status
            while True:
                # Check if session still exists
                check_result = subprocess.run(
                    ["tmux", "has-session", "-t", session_name],
                    capture_output=True
                )
                
                if check_result.returncode != 0:
                    # Session ended, try to get the exit status
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
                # Safety timeout check
                if time.time() - start_time > 7200:  # 2 hour timeout
                    subprocess.run(["tmux", "kill-session", "-t", session_name], 
                                 capture_output=True, check=False)
                    logger.error(f"[{device_id}] ⏰ Timeout {config_name}")
                    return config_name, -1, f"Timeout on {device_id} after 2 hours"
            
            duration = time.time() - start_time
            
            # Try to capture any output from the tmux session
            capture_result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True, text=True, check=False
            )
            
            # Since we can't easily get the exit code from tmux, assume success if it completed
            # without timeout and the process finished normally
            logger.info(f"[{device_id}] ✅ Completed {config_name} in {duration:.1f}s")
            
            if capture_result.returncode == 0 and capture_result.stdout:
                # Show last few lines of output
                output_lines = capture_result.stdout.strip().split('\n')
                logger.info(f"[{device_id}] Final output: {output_lines[-1] if output_lines else 'No output'}")
                
                # Check for obvious error patterns in the output
                full_output = capture_result.stdout.lower()
                if any(error_word in full_output for error_word in ['error', 'exception', 'traceback', 'failed']):
                    logger.error(f"[{device_id}] Detected errors in output")
                    return config_name, 1, f"Errors detected in output on {device_id}"
            
            return config_name, 0, f"Success on {device_id} in {duration:.1f}s (tmux: {session_name})"
            
        else:
                        # Original subprocess approach (fallback)
            cmd = ["python", "run.py", "--config_path", str(config_path), "--device", device_id]
            logger.info(f"[{device_id}] Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"[{device_id}] ✅ Completed {config_name} in {duration:.1f}s")
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')[:3]
                    logger.info(f"[{device_id}] Stdout preview: {' | '.join(stdout_lines)}")
                return config_name, 0, f"Success on {device_id} in {duration:.1f}s"
            else:
                logger.error(f"[{device_id}] ❌ Failed {config_name} (return code: {result.returncode})")
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    for line in stderr_lines[-3:]:
                        logger.error(f"[{device_id}] STDERR: {line}")
                
                error_msg = result.stderr.strip()[:200] if result.stderr else f"Return code: {result.returncode}"
                return config_name, result.returncode, f"Failed on {device_id}: {error_msg}"
            
    except subprocess.TimeoutExpired:
        logger.error(f"[{device_id}] ⏰ Timeout {config_name}")
        return config_name, -1, f"Timeout on {device_id} after 2 hours"
    except Exception as e:
        logger.error(f"[{device_id}] 💥 Error {config_name}: {e}")
        return config_name, -2, f"Error on {device_id}: {e}"


def validate_cuda_devices(devices: List[str]) -> None:
    """Validate that all CUDA devices exist on the system."""
    if not torch.cuda.is_available():
        cuda_devices = [d for d in devices if d.startswith('cuda')]
        if cuda_devices:
            raise RuntimeError("CUDA devices specified but CUDA is not available on this system")
        return
    
    num_devices = torch.cuda.device_count()
    available_devices = [f"cuda:{i}" for i in range(num_devices)]
    
    for device in devices:
        if device.startswith('cuda'):
            if ':' in device:
                try:
                    device_index = int(device.split(':')[1])
                except ValueError:
                    raise RuntimeError(f"Invalid device format: {device}")
            else:
                device_index = 0
            
            if device_index >= num_devices:
                raise RuntimeError(
                    f"Invalid CUDA device {device}. "
                    f"System has {num_devices} CUDA device(s): {available_devices}"
                )
    
    logger.info(f"Validated devices: {devices}")


def list_available_devices() -> List[str]:
    """List all available CUDA devices on the system."""
    devices = ['cpu']
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devices.extend([f"cuda:{i}" for i in range(num_devices)])
    return devices


def run_experiments_on_devices(config_paths: List[Path], device_ids: List[str], sequential: bool = False, use_tmux: bool = True) -> List[tuple[str, int, str]]:
    """Run experiments distributed across multiple devices."""
    # Validate all devices before starting any experiments
    validate_cuda_devices(device_ids)
    
    results = []
    
    # Create device-config pairs by cycling through devices
    device_config_pairs = []
    for i, config_path in enumerate(config_paths):
        device_id = device_ids[i % len(device_ids)]
        device_config_pairs.append((config_path, device_id))
    
    # Log distribution
    device_counts = {device_id: 0 for device_id in device_ids}
    for _, device_id in device_config_pairs:
        device_counts[device_id] += 1
    
    logger.info("Experiment distribution across devices:")
    for device_id, count in device_counts.items():
        logger.info(f"  {device_id}: {count} experiments")
    
    if sequential:
        # Run experiments sequentially for debugging
        logger.info("Running experiments sequentially (debugging mode)")
        for i, (config_path, device_id) in enumerate(device_config_pairs):
            logger.info(f"Running experiment {i+1}/{len(device_config_pairs)}: {config_path.name} on {device_id}")
            result = run_single_experiment(config_path, device_id, use_tmux=use_tmux)
            results.append(result)
            logger.info(f"Completed: {result[0]} ({len(results)}/{len(config_paths)})")
    else:
        # Run experiments in parallel
        logger.info("Running experiments in parallel")
        with ProcessPoolExecutor(max_workers=len(device_ids)) as executor:
            # Submit all jobs with their assigned devices
            future_to_info = {
                executor.submit(run_single_experiment, config_path, device_id, use_tmux): (config_path, device_id)
                for config_path, device_id in device_config_pairs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_info):
                config_path, device_id = future_to_info[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result[0]} ({len(results)}/{len(config_paths)})")
                except Exception as e:
                    error_result = (config_path.name, -3, f"Process error on {device_id}: {e}")
                    results.append(error_result)
                    logger.error(f"Process failed: {config_path.name} on {device_id}: {e}")
    
    return results


def print_experiment_summary(results: List[tuple[str, int, str]], device_ids: List[str], total_duration: float):
    """Print a comprehensive summary of experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Device-specific performance
    print(f"\nDevice-specific performance:")
    device_results = {}
    for config_name, return_code, message in results:
        # Extract device from message
        if " on " in message:
            device = message.split(" on ")[1].split(" ")[0]
            if device not in device_results:
                device_results[device] = {"success": 0, "failed": 0}
            if return_code == 0:
                device_results[device]["success"] += 1
            else:
                device_results[device]["failed"] += 1
    
    for device, stats in device_results.items():
        total = stats["success"] + stats["failed"]
        success_rate = (stats["success"] / total * 100) if total > 0 else 0
        print(f"  {device}: {stats['success']}/{total} success ({success_rate:.1f}%)")
    
    print(f"\nTiming:")
    print(f"Total sweep duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Average time per experiment: {total_duration/len(results):.1f}s")
    
    if len(device_ids) > 1:
        theoretical_sequential_time = len(device_ids) * (total_duration / len(device_ids))
        speedup = theoretical_sequential_time / total_duration
        print(f"Estimated speedup from multi-device: {speedup:.1f}x")
    
    # Show active tmux sessions
    try:
        tmux_result = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True, check=False)
        if tmux_result.returncode == 0 and tmux_result.stdout:
            active_sessions = [line for line in tmux_result.stdout.strip().split('\n') if 'exp_' in line]
            if active_sessions:
                print(f"\nActive experiment tmux sessions:")
                for session in active_sessions:
                    session_name = session.split(':')[0]
                    print(f"  tmux attach -t {session_name}")
    except Exception:
        pass  # tmux not available
    
    if failed:
        print("\nFailed Experiments:")
        for config_name, return_code, message in failed[:10]:  # Show first 10 failures
            print(f"  ❌ {config_name}: {message}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="SAE experiment sweep with separated config and sweep definitions")
    parser.add_argument("--base_config", type=str, required=True,
                        help="Base configuration file (e.g., configs/tinystories-gated.yaml)")
    parser.add_argument("--sweep_config", type=str, required=True,
                        help="Sweep configuration file (e.g., sweep_configs/default_sweep.yaml)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda:0', 'cuda:1', 'cpu', etc.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show experiment configurations and device distribution without running")
    parser.add_argument("--show_devices", action="store_true",
                        help="Just show available devices and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override max_experiments from sweep config")

    args = parser.parse_args()
    
    # Show devices if requested
    if args.show_devices:
        devices = detect_available_devices()
        print("Available devices:")
        for device in devices:
            print(f"  {device}")
        return
    
    # Validate input files
    base_config_path = Path(args.base_config)
    sweep_config_path = Path(args.sweep_config)
    
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    if not sweep_config_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {sweep_config_path}")
    
    # Load configurations
    logger.info(f"Loading base config from: {base_config_path}")
    base_config = load_base_config(base_config_path)
    
    logger.info(f"Loading sweep config from: {sweep_config_path}")
    sweep_config = load_sweep_config(sweep_config_path)
    
    # Detect SAE type and generate parameter combinations
    sae_type = detect_sae_type(base_config)
    logger.info(f"Detected SAE type: {sae_type}")
    
    param_combinations = generate_parameter_combinations(sweep_config)
    
    # Apply limit override if specified
    if args.limit and len(param_combinations) > args.limit:
        logger.info(f"Limiting to {args.limit} experiments (from {len(param_combinations)} combinations)")
        param_combinations = param_combinations[:args.limit]
    
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    
    # Generate experiment configurations
    experiment_configs = []
    for i, params in enumerate(param_combinations):
        config = create_experiment_config(base_config, params, i)
        experiment_configs.append(config)

    if args.dry_run:
        print(f"\nDevice distribution (dry run):")
        print(f"  {args.device}: {len(experiment_configs)} experiments")
        
        print(f"\nAll experiment configurations:")
        for i, config in enumerate(experiment_configs):
            params = config.get('_params', {})
            # Format parameters nicely for display
            param_strs = []
            for k, v in params.items():
                if isinstance(v, float):
                    if v < 0.01:
                        param_strs.append(f"{k}={v:.0e}")
                    else:
                        param_strs.append(f"{k}={v}")
                else:
                    param_strs.append(f"{k}={v}")
            print(f"  Experiment {i}: {', '.join(param_strs)}")
        
        return

    if args.device:
        device = torch.device(args.device)
        if device.type == 'cuda':
            # Validate CUDA device exists
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            num_devices = torch.cuda.device_count()
            if device.index is None:
                device_index = 0
            else:
                device_index = device.index
                
            if device_index >= num_devices:
                available_devices = [f"cuda:{i}" for i in range(num_devices)]
                raise RuntimeError(
                    f"Invalid CUDA device {args.device}. "
                    f"System has {num_devices} CUDA device(s): {available_devices}"
                )
            print(f"Set CUDA device context to: {device} (validated)")
    else:
        # Default to cuda:0 if available, but be explicit about it  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run experiments
    start_time = time.time()
    for i, config in enumerate(experiment_configs):
        # Remove metadata before running
        clean_config = {k: v for k, v in config.items() if not k.startswith('_')}
        
        logger.info(f"Running experiment {i+1}/{len(experiment_configs)}")
        # Convert device string to torch.device object if specified
        run(clean_config, device=device)
        
    total_duration = time.time() - start_time
    logger.info(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

if __name__ == "__main__":
    main()