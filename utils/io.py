import yaml
from pathlib import Path
from typing import Any
import os
import tempfile
import json

import torch
import wandb
from torch import nn
from utils.logging import logger
from utils.types import BaseModelType
from utils.constants import CONFIG_FILE


def save_module(
    config_dict: dict[str, Any],
    save_dir: Path,
    module: nn.Module,
    model_filename: str,
    config_filename: str = CONFIG_FILE,
) -> None:
    """Save the pytorch module and config to the save_dir.

    The config will only be saved if the save_dir doesn't exist (i.e. the first time the module is
    saved assuming the save_dir is unique to the module).

    Args:
        config_dict: Dictionary representation of the config to save.
        save_dir: Directory to save the module.
        module: The module to save.
        model_filename: The filename to save the model to.
        config_filename: The filename to save the config to.
    """
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        with open(save_dir / config_filename, "w") as f:
            yaml.dump(config_dict, f)
        logger.info("Saved config to %s", save_dir / config_filename)

    torch.save(module.state_dict(), save_dir / model_filename)
    logger.info("Saved model to %s", save_dir / model_filename)


def load_config(config_path_or_obj: Path | str | BaseModelType | dict[str, Any], config_model: type[BaseModelType]) -> BaseModelType:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)
    
    if isinstance(config_path_or_obj, dict):
        config_dict = config_path_or_obj
    else:
        assert isinstance(
            config_path_or_obj, Path
        ), f"passed config is of invalid type {type(config_path_or_obj)}"
        assert (
            config_path_or_obj.suffix == ".yaml"
        ), f"Config file {config_path_or_obj} must be a YAML file."
        assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
        with open(config_path_or_obj) as f:
            config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


# File Upload Functions
def save_activation_data_to_wandb(
    accumulated_data: dict[str, dict[str, torch.Tensor]], 
    all_token_ids: list[list[str]] | None = None,
    output_path: str = "./artifacts",
    skip_upload: bool = False,
    chunk_upload: bool = True
) -> None:
    """Save accumulated activation data as Wandb artifacts to the current run.
    
    Args:
        accumulated_data: Dictionary mapping SAE positions to activation data
        all_token_ids: Optional list of token ID sequences to save alongside activation data
        output_path: Path for storing temporary files (default: ./artifacts)
        skip_upload: If True, only save locally without uploading to Wandb (default: False)
        chunk_upload: If True, upload each file individually and clean up immediately (default: True)
    """
    # Ensure we have an active run (unless we're skipping upload)
    if not skip_upload:
        assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
        
        # Set Wandb's cache directories to use our output_path
        import os
        output_path_abs = Path(output_path).absolute()
        wandb_cache_dir = output_path_abs / "wandb_cache"
        wandb_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for Wandb to use our directory
        os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(wandb_cache_dir)
        os.environ["WANDB_DIR"] = str(wandb_cache_dir)
        
        print(f"Set Wandb cache directory to: {wandb_cache_dir}")
    
    # Check available disk space in the output path
    import shutil
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stat = shutil.disk_usage(str(output_path))
    available_gb = stat.free / (1024**3)
    
    # Estimate size needed
    total_elements = 0
    for data in accumulated_data.values():
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                total_elements += value.numel()
    # float16 = 2 bytes, int64 = 8 bytes, estimate average of 4 bytes
    estimated_gb = (total_elements * 4) / (1024**3)
    
    print(f"Output path: {output_path.absolute()}")
    print(f"Available disk space: {available_gb:.2f} GB")
    print(f"Estimated data size: {estimated_gb:.2f} GB")
    
    # For chunked upload, we only need space for one file at a time
    if chunk_upload and not skip_upload:
        # Find the largest single file size
        max_file_gb = 0
        for data in accumulated_data.values():
            file_size_gb = sum(
                v.numel() * v.element_size() / (1024**3) 
                for v in data.values() 
                if isinstance(v, torch.Tensor)
            )
            max_file_gb = max(max_file_gb, file_size_gb)
        
        space_needed = max_file_gb * 2  # Need space for file + Wandb's cache
        if available_gb < space_needed:
            print(f"WARNING: May not have enough disk space even for chunked upload!")
            print(f"  Available: {available_gb:.2f} GB, Need per chunk: ~{space_needed:.2f} GB")
    else:
        # For non-chunked upload or skip_upload, check total space
        space_needed_for_upload = estimated_gb * 2.5
        if not skip_upload and available_gb < space_needed_for_upload:
            print(f"WARNING: May not have enough disk space for full upload!")
            print(f"  Available: {available_gb:.2f} GB, Need for upload: ~{space_needed_for_upload:.2f} GB")
            print(f"  Consider using chunked upload (default) or --skip_upload flag")
    
    # Create a unique subdirectory for this run
    run_id = wandb.run.id if wandb.run else "local"
    run_name = wandb.run.name if wandb.run else "local"
    clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
    
    activation_data_dir = output_path / f"activation_data_{run_id}"
    activation_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using local directory for staging: {activation_data_dir}")
    
    if skip_upload:
        # Save all files locally without uploading
        print("Saving all files locally (skip_upload=True)...")
        _save_all_files_locally(accumulated_data, all_token_ids, activation_data_dir, output_path)
        print(f"Files saved locally at: {activation_data_dir}")
    elif chunk_upload and not skip_upload:
        # Chunked upload: save and upload one file at a time
        print("Using chunked upload strategy to minimize disk usage...")
        chunk_idx = 0
        total_chunks = len(accumulated_data) + (1 if all_token_ids is not None else 0)
        
        # Upload each SAE position's data as a separate chunk
        for sae_pos, data in accumulated_data.items():
            chunk_idx += 1
            print(f"\n[Chunk {chunk_idx}/{total_chunks}] Processing {sae_pos}...")
            
            safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
            file_path = activation_data_dir / f"{safe_layer_name}.pt"
            
            # Make tensors contiguous before saving
            contiguous_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    contiguous_data[key] = value.contiguous()
                else:
                    contiguous_data[key] = value
            
            # Save the chunk
            tensor_size_gb = sum(
                v.numel() * v.element_size() / (1024**3) 
                for v in contiguous_data.values() 
                if isinstance(v, torch.Tensor)
            )
            
            try:
                torch.save(contiguous_data, file_path)
                print(f"  Saved {file_path.name} ({tensor_size_gb:.2f} GB)")
                
                # Create and upload artifact for this chunk
                artifact_name = f"activation_data_{clean_run_name}_{run_id}_chunk_{chunk_idx:03d}_{safe_layer_name}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="activation_data_chunk",
                    description=f"SAE activation data chunk {chunk_idx}/{total_chunks} for {sae_pos}"
                )
                artifact.add_file(str(file_path), name=f"{safe_layer_name}.pt")
                
                print(f"  Uploading chunk to Wandb...")
                wandb.log_artifact(artifact)
                print(f"  Successfully uploaded chunk: {artifact_name}")
                
                # Clean up the chunk immediately after upload
                file_path.unlink()
                print(f"  Cleaned up local chunk file")
                
            except Exception as e:
                print(f"  ERROR: Failed to process chunk {sae_pos}: {e}")
                # Keep the file if upload failed
                print(f"  Keeping failed chunk at: {file_path}")
        
        # Upload token IDs if provided
        if all_token_ids is not None:
            chunk_idx += 1
            print(f"\n[Chunk {chunk_idx}/{total_chunks}] Processing token IDs...")
            
            token_ids_path = activation_data_dir / "all_token_ids.pt"
            try:
                torch.save(all_token_ids, token_ids_path)
                file_size_gb = token_ids_path.stat().st_size / (1024**3)
                print(f"  Saved all_token_ids.pt ({file_size_gb:.2f} GB)")
                
                # Create and upload artifact for token IDs
                artifact_name = f"activation_data_{clean_run_name}_{run_id}_chunk_{chunk_idx:03d}_token_ids"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="activation_data_chunk",
                    description=f"Token IDs chunk {chunk_idx}/{total_chunks}"
                )
                artifact.add_file(str(token_ids_path), name="all_token_ids.pt")
                
                print(f"  Uploading chunk to Wandb...")
                wandb.log_artifact(artifact)
                print(f"  Successfully uploaded chunk: {artifact_name}")
                
                # Clean up
                token_ids_path.unlink()
                print(f"  Cleaned up local chunk file")
                
            except Exception as e:
                print(f"  ERROR: Failed to process token IDs: {e}")
                print(f"  Keeping failed chunk at: {token_ids_path}")
        
        # Clean up the staging directory if it's empty
        try:
            if not any(activation_data_dir.iterdir()):
                activation_data_dir.rmdir()
                print(f"\nCleaned up empty staging directory: {activation_data_dir}")
            else:
                print(f"\nSome files remain at: {activation_data_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up directory {activation_data_dir}: {e}")
        
        print(f"\nChunked upload complete! Uploaded {total_chunks} chunks.")
        
    else:
        # Original non-chunked upload (saves all files then uploads together)
        print("Using traditional upload (all files at once)...")
        _save_all_files_locally(accumulated_data, all_token_ids, activation_data_dir, output_path)
        
        try:
            # Create single artifact for all data
            artifact_name = f"evaluation_activation_data_{clean_run_name}_{run_id}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="activation_data",
                description=f"SAE activation data for run {run_name} ({run_id})"
            )
            
            # Add all files
            print(f"Uploading all files to Wandb...")
            for file_path in activation_data_dir.glob("*.pt"):
                file_size_gb = file_path.stat().st_size / (1024**3)
                print(f"  Adding {file_path.name} ({file_size_gb:.2f} GB) to artifact...")
                artifact.add_file(str(file_path), name=f"activation_data/{file_path.name}")
            
            # Log the artifact
            print("Logging artifact to Wandb...")
            wandb.log_artifact(artifact, aliases=["latest"])
            print(f"Successfully uploaded activation data as Wandb artifact: {artifact_name}")
            
            # Clean up all files
            shutil.rmtree(activation_data_dir)
            print(f"Cleaned up staging directory: {activation_data_dir}")
            
        except Exception as e:
            print(f"Warning: Failed to upload activation data artifact to Wandb: {e}")
            print(f"Local files remain at: {activation_data_dir}")


def _save_all_files_locally(accumulated_data, all_token_ids, activation_data_dir, output_path):
    """Helper function to save all files locally."""
    import shutil
    
    for sae_pos, data in accumulated_data.items():
        safe_layer_name = sae_pos.replace(".", "--").replace("/", "--")
        file_path = activation_data_dir / f"{safe_layer_name}.pt"
        
        # Make tensors contiguous before saving
        contiguous_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                contiguous_data[key] = value.contiguous()
            else:
                contiguous_data[key] = value
        
        # Check disk space before each save
        stat = shutil.disk_usage(str(output_path))
        available_gb = stat.free / (1024**3)
        
        # Estimate size for this specific tensor
        tensor_size_gb = sum(
            v.numel() * v.element_size() / (1024**3) 
            for v in contiguous_data.values() 
            if isinstance(v, torch.Tensor)
        )
        
        if available_gb < tensor_size_gb * 1.2:
            raise RuntimeError(f"Insufficient disk space for {sae_pos}. Need ~{tensor_size_gb:.2f} GB, have {available_gb:.2f} GB")
        
        torch.save(contiguous_data, file_path)
        print(f"Staged activation data for {sae_pos} at {file_path} ({tensor_size_gb:.2f} GB)")
    
    # Save token IDs if provided
    if all_token_ids is not None:
        token_ids_path = activation_data_dir / "all_token_ids.pt"
        torch.save(all_token_ids, token_ids_path)
        file_size_gb = token_ids_path.stat().st_size / (1024**3)
        print(f"Staged token IDs at {token_ids_path} ({file_size_gb:.2f} GB)")


def save_metrics_to_wandb(
    metrics: dict[str, dict[str, Any]],
    output_path: str = "/tmp"
) -> None:
    """Save evaluation metrics as a Wandb artifact to the current run.
    
    Args:
        metrics: Dictionary mapping SAE positions to their evaluation metrics
        output_path: Path for storing temporary files (default: /tmp)
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    
    # Use a temporary file for metrics in the specified output path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='metrics_', dir=str(output_path)) as f:
        json.dump(metrics, f, indent=2)
        temp_metrics_path = f.name
    
    print(f"Staged metrics at {temp_metrics_path}")
    
    try:
        # Create artifact name with run name and ID for easy identification
        run_name = wandb.run.name or "unnamed"
        run_id = wandb.run.id
        # Clean run name for artifact naming (replace invalid characters)
        clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        artifact_name = f"evaluation_metrics_{clean_run_name}_{run_id}"
        
        # Create artifact for metrics with alias to override existing
        artifact = wandb.Artifact(
            name=artifact_name,
            type="metrics",
            description=f"SAE evaluation metrics for run {run_name} ({run_id})"
        )
        artifact.add_file(temp_metrics_path, name="metrics.json")
        
        # Log the artifact with "latest" alias to override previous versions
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f"Successfully uploaded metrics as Wandb artifact: {artifact_name}")
    except Exception as e:
        print(f"Warning: Failed to upload metrics artifact to Wandb: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_metrics_path)
        except:
            pass


# Local file functions removed - now only using Wandb artifacts


def load_activation_data_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m",
    output_path: str = "./artifacts",
    use_cached: bool = True,
    parallel_download: bool = True,
    max_workers: int = 4
) -> tuple[dict[str, dict[str, torch.Tensor]], list[list[str]] | None]:
    """Load accumulated activation data and token IDs from local files or Wandb artifacts.
    
    Checks for local files in the following order:
    1. activation_data_{run_id}/ - Created when using skip_upload=True
    2. cached_{run_id}/ - Created from previous Wandb downloads
    3. Wandb artifacts - Downloads from Wandb if no local files found
    
    Supports both single artifact format and chunked artifact format.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
        output_path: Path for downloading artifacts (default: ./artifacts)
        use_cached: If True, use locally cached files if they exist (default: True)
        parallel_download: If True, download chunks in parallel (default: True)
        max_workers: Maximum number of parallel downloads (default: 4)
    
    Returns:
        Tuple of (activation_data_dict, all_token_ids)
        
    Raises:
        FileNotFoundError: If the run or files are not found
        RuntimeError: If the files exist but can't be loaded
    """
    try:
        # Check for local files first
        output_path = Path(output_path)
        
        # Check for local activation data directory (created when skip_upload=True)
        local_activation_dir = output_path / f"activation_data_{run_id}"
        local_cache_dir = output_path / f"cached_{run_id}"
        
        # Try local activation data directory first, then cached directory
        for local_dir in [local_activation_dir, local_cache_dir]:
            if use_cached and local_dir.exists():
                print(f"Found local data at: {local_dir}")
                print("Loading from local files (skip download)...")
                
                accumulated_data: dict[str, dict[str, torch.Tensor]] = {}
                all_token_ids: list[list[str]] | None = None
                
                # Load from local directory
                for file_path in local_dir.glob("*.pt"):
                    filename = file_path.name
                    if filename == "all_token_ids.pt":
                        all_token_ids = torch.load(file_path, map_location='cpu')
                        print(f"  Loaded token IDs from local")
                    else:
                        safe_layer_name = filename[:-3]
                        sae_pos = safe_layer_name.replace("--", ".")
                        data = torch.load(file_path, map_location='cpu')
                        accumulated_data[sae_pos] = data
                        print(f"  Loaded {sae_pos} from local")
                
                if accumulated_data:
                    print(f"Successfully loaded {len(accumulated_data)} SAE positions from local files")
                    return accumulated_data, all_token_ids
        
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            raise FileNotFoundError(f"Run {run_id} not found in project {project}")
        
        # Get all logged artifacts from this run
        artifacts = list(run.logged_artifacts())
        
        # First, try to load from chunked artifacts (new format)
        try:
            chunk_artifacts = [a for a in artifacts if a.type == "activation_data_chunk" and run_id in a.name]
            
            if chunk_artifacts:
                print(f"Found {len(chunk_artifacts)} chunked activation data artifacts")
                
                # Create cache directory
                local_cache_dir.mkdir(parents=True, exist_ok=True)
                
                accumulated_data: dict[str, dict[str, torch.Tensor]] = {}
                all_token_ids: list[list[str]] | None = None
                
                if parallel_download and len(chunk_artifacts) > 1:
                    # Parallel download using ThreadPoolExecutor
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                
                def download_chunk(chunk_artifact, idx):
                    """Download a single chunk."""
                    print(f"Starting download of chunk {idx}/{len(chunk_artifacts)}: {chunk_artifact.name.split('_')[-1]}")
                    
                    # Download to temp directory
                    temp_dir = output_path / f"temp_chunk_{idx:03d}_{run_id}"
                    chunk_dir = chunk_artifact.download(root=str(temp_dir))
                    
                    # Process the chunk
                    result_data = {}
                    result_token_ids = None
                    
                    for file_path in Path(chunk_dir).glob("*.pt"):
                        filename = file_path.name
                        cache_path = local_cache_dir / filename
                        
                        # Copy to cache
                        import shutil
                        shutil.copy2(file_path, cache_path)
                        
                        if filename == "all_token_ids.pt":
                            result_token_ids = torch.load(cache_path, map_location='cpu')
                            print(f"  Downloaded token IDs")
                        else:
                            safe_layer_name = filename[:-3]
                            sae_pos = safe_layer_name.replace("--", ".")
                            data = torch.load(cache_path, map_location='cpu')
                            result_data[sae_pos] = data
                            print(f"  Downloaded {sae_pos}")
                    
                    # Clean up temp directory
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                    
                    return result_data, result_token_ids
                
                print(f"Downloading {len(chunk_artifacts)} chunks in parallel (max {max_workers} workers)...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all download tasks
                    future_to_idx = {
                        executor.submit(download_chunk, chunk, i): i 
                        for i, chunk in enumerate(chunk_artifacts, 1)
                    }
                    
                    # Process completed downloads
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            chunk_data, chunk_token_ids = future.result()
                            accumulated_data.update(chunk_data)
                            if chunk_token_ids is not None:
                                all_token_ids = chunk_token_ids
                        except Exception as e:
                            print(f"Error downloading chunk {idx}: {e}")
                
            else:
                # Sequential download (original method)
                for i, chunk_artifact in enumerate(chunk_artifacts, 1):
                    print(f"Downloading chunk {i}/{len(chunk_artifacts)}: {chunk_artifact.name}")
                    
                    # Download to temp directory
                    temp_dir = output_path / f"temp_chunk_{i:03d}_{run_id}"
                    chunk_dir = chunk_artifact.download(root=str(temp_dir))
                    
                    # Copy files to cache and load
                    for file_path in Path(chunk_dir).glob("*.pt"):
                        filename = file_path.name
                        cache_path = local_cache_dir / filename
                        
                        # Copy to cache
                        import shutil
                        shutil.copy2(file_path, cache_path)
                        
                        if filename == "all_token_ids.pt":
                            all_token_ids = torch.load(cache_path, map_location='cpu')
                            print(f"  Loaded token IDs from chunk")
                        else:
                            safe_layer_name = filename[:-3]
                            sae_pos = safe_layer_name.replace("--", ".")
                            data = torch.load(cache_path, map_location='cpu')
                            accumulated_data[sae_pos] = data
                            print(f"  Loaded activation data for {sae_pos}")
                    
                    # Clean up temp directory
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
            
            if accumulated_data:
                print(f"Successfully loaded {len(accumulated_data)} SAE positions from chunks")
                print(f"Data cached at: {local_cache_dir}")
                return accumulated_data, all_token_ids
        except Exception as e:
            print(f"Could not load from chunked artifacts: {e}")
        
        # Fall back to loading from single artifact (old format)
        try:
            activation_artifacts = [a for a in artifacts if a.type == "activation_data" and "evaluation_activation_data" in a.name]
            
            if not activation_artifacts:
                raise FileNotFoundError(f"No activation data files found in run {run_id}")
            
            # Use the latest activation data artifact (highest version)
            latest_artifact = max(activation_artifacts, key=lambda x: x.version)
            
            # Download to the specified output path
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            download_dir = output_path / f"downloaded_{run_id}"
            
            print(f"Downloading artifact to: {download_dir}")
            artifact_dir = latest_artifact.download(root=str(download_dir))
            
            activation_data_path = Path(artifact_dir) / "activation_data"
            if not activation_data_path.exists():
                raise FileNotFoundError(f"Activation data directory missing in artifact for run {run_id}")
            
            accumulated_data: dict[str, dict[str, torch.Tensor]] = {}
            all_token_ids: list[list[str]] | None = None
            
            # Load activation data files
            for file_path in activation_data_path.glob("*.pt"):
                filename = file_path.name
                if filename == "all_token_ids.pt":
                    # Load token IDs
                    all_token_ids = torch.load(file_path, map_location='cpu')
                    print(f"Loaded token IDs from Wandb artifact")
                else:
                    # Convert filename back to original sae_pos format
                    safe_layer_name = filename[:-3]  # Remove .pt extension
                    sae_pos = safe_layer_name.replace("--", ".")  # Convert back from safe filename
                    data = torch.load(file_path, map_location='cpu')  # Load to CPU first
                    accumulated_data[sae_pos] = data
                    print(f"Loaded activation data for {sae_pos} from Wandb artifact")
            
            if not accumulated_data:
                raise FileNotFoundError(f"No activation data tensors found in artifact for run {run_id}")
            
            print(f"Loaded activation data from Wandb artifact: {latest_artifact.name} (v{latest_artifact.version})")
            
            # Clean up downloaded files after loading to memory
            try:
                import shutil
                shutil.rmtree(download_dir)
                print(f"Cleaned up downloaded files from: {download_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up downloaded files at {download_dir}: {e}")
            
            return accumulated_data, all_token_ids
        except Exception as e:
            print(f"Could not load activation data from artifacts: {e}")
            raise FileNotFoundError(f"No activation data files found in run {run_id}")
    except FileNotFoundError:
        # Re-raise these specific exceptions
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading activation data from Wandb: {e}")


def load_metrics_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m",
    output_path: str = "./artifacts",
    use_logged_metrics: bool = True
) -> dict[str, dict[str, Any]] | None:
    """Load evaluation metrics from Wandb artifacts or logged metrics.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
        output_path: Path for downloading artifacts (default: ./artifacts)
        use_logged_metrics: If True, try to load from logged metrics (eval/...) first (default: True)
    
    Returns:
        Dictionary mapping SAE positions to their evaluation metrics, or None if not found
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            print(f"Run {run_id} not found in project {project}")
            return None
        
        # Try to load from logged metrics first (eval/... keys in summary or history)
        if use_logged_metrics:
            try:
                # Check run summary for eval metrics
                summary = run.summary
                
                # Look for eval metrics in summary (format: eval/{metric_name}/{layer_name})
                eval_metrics: dict[str, dict[str, Any]] = {}
                
                # Metric name mapping from logged format to expected format
                # Only include the metrics we care about (not loss-related ones)
                metric_name_mapping = {
                    'L_0': 'sparsity_l0',
                    'mse': 'mse',
                    'explained_variance': 'explained_variance',
                    'explained_variance_ln': 'explained_variance_ln',
                    'alive_dict_components': 'alive_dict_components',
                    'alive_dict_components_proportion': 'alive_dict_components_proportion',
                }
                
                for key, value in summary.items():
                    if key.startswith("eval/"):
                        # Parse key: eval/{metric_name}/{layer_name}
                        parts = key.split("/")
                        if len(parts) == 3:  # Only exact format: eval/metric/layer
                            metric_name = parts[1]
                            layer_name = parts[2]
                            
                            # Only process metrics we care about
                            if metric_name in metric_name_mapping:
                                mapped_metric = metric_name_mapping[metric_name]
                                
                                if layer_name not in eval_metrics:
                                    eval_metrics[layer_name] = {}
                                eval_metrics[layer_name][mapped_metric] = value
                
                if eval_metrics:
                    print(f"Loaded evaluation metrics from logged data (eval/...) for {len(eval_metrics)} layers")
                    return eval_metrics
                    
            except Exception as e:
                print(f"Could not load from logged metrics: {e}")
        
        # Fall back to loading from artifacts
        try:
            # Get all logged artifacts from this run
            artifacts = list(run.logged_artifacts())
            metrics_artifacts = [a for a in artifacts if a.type == "metrics" and "evaluation_metrics" in a.name]
            
            if not metrics_artifacts:
                print(f"No metrics found in run {run_id}")
                return None
            
            # Use the latest metrics artifact (highest version)
            latest_artifact = max(metrics_artifacts, key=lambda x: x.version)
            
            # Download to the specified output path
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            download_dir = output_path / f"downloaded_metrics_{run_id}"
            
            artifact_dir = latest_artifact.download(root=str(download_dir))
            
            metrics_path = Path(artifact_dir) / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                print(f"Loaded evaluation metrics from Wandb artifact: {latest_artifact.name} (v{latest_artifact.version})")
                
                # Clean up downloaded files after loading
                try:
                    import shutil
                    shutil.rmtree(download_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up downloaded files at {download_dir}: {e}")
                
                return metrics
        except Exception as e:
            print(f"Could not load metrics from artifacts: {e}")
            return None
        
        print(f"No metrics found in run {run_id}")
        return None
    except Exception as e:
        print(f"Error loading metrics from Wandb: {e}")
        return None


# Evaluation Results Functions

def save_explanations_to_wandb(
    explanations: dict[str, dict[str, Any]],
    output_path: str = "/tmp"
) -> None:
    """Save explanations as Wandb artifacts to the current run.
    
    Args:
        explanations: Dictionary mapping neuron keys to explanation data
        output_path: Path for storing temporary files (default: /tmp)
    """
    assert wandb.run is not None, "No active Weights & Biases run. Call wandb.init() first."
    
    # Use a temporary file for explanations in the specified output path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='explanations_', dir=str(output_path)) as f:
        json.dump(explanations, f, indent=2)
        temp_explanations_path = f.name
    
    # Create summary statistics
    summary_stats = {
        "num_explanations": len(explanations),
        "explained_neurons_per_layer": {}
    }
    
    # Handle both nested dict structure (SAE positions -> neurons) and flat structure
    for key, value in explanations.items():
        if isinstance(value, dict) and not any(k in ['explanation', 'detection_score', 'fuzz_score'] for k in value.keys()):
            # Nested structure: key is SAE position, value is dict of neurons
            layer_name = key
            summary_stats["explained_neurons_per_layer"][layer_name] = len(value)
        elif "_neuron_" in key:
            # Flat structure: key contains neuron info
            layer_name = key.split("_neuron_")[0]
            if layer_name not in summary_stats["explained_neurons_per_layer"]:
                summary_stats["explained_neurons_per_layer"][layer_name] = 0
            summary_stats["explained_neurons_per_layer"][layer_name] += 1
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='explanation_summary_', dir=str(output_path)) as f:
        json.dump(summary_stats, f, indent=2)
        temp_summary_path = f.name

    print(f"Staged explanations at {temp_explanations_path}")
    print(f"Staged explanation summary at {temp_summary_path}")
    
    try:
        # Create artifact name with run name and ID for easy identification
        run_name = wandb.run.name or "unnamed"
        run_id = wandb.run.id
        # Clean run name for artifact naming (replace invalid characters)
        clean_run_name = run_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        artifact_name = f"evaluation_explanations_{clean_run_name}_{run_id}"
        
        # Create artifact for explanations with alias to override existing
        artifact = wandb.Artifact(
            name=artifact_name,
            type="explanations",
            description=f"Neuron explanations for run {run_name} ({run_id})"
        )
        artifact.add_file(temp_explanations_path, name="explanations.json")
        artifact.add_file(temp_summary_path, name="explanation_summary.json")
        
        # Log the artifact with "latest" alias to override previous versions
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f"Successfully uploaded explanations as Wandb artifact: {artifact_name}")
    except Exception as e:
        print(f"Warning: Failed to upload explanations artifact to Wandb: {e}")
    finally:
        # Clean up temp files
        try:
            os.unlink(temp_explanations_path)
            os.unlink(temp_summary_path)
        except:
            pass


def load_explanations_from_wandb(
    run_id: str,
    project: str = "raymondl/tinystories-1m",
    output_path: str = "./artifacts",
    version: int | str | None = None
) -> dict[str, dict[str, Any]] | None:
    """Load explanations from Wandb artifacts.
    
    Args:
        run_id: The Wandb run ID to load files from
        project: Wandb project name
        output_path: Path for downloading artifacts (default: ./artifacts)
        version: Specific artifact version to load. Can be:
                 - None (default): uses latest version
                 - int: specific version number (e.g., 2)
                 - str: version string (e.g., "v2" or "latest")
    
    Returns:
        Dictionary mapping neuron keys to explanation data, or None if not found
    """
    try:
        # Initialize Wandb API
        api = wandb.Api()
        
        # Get the run
        try:
            run = api.run(f"{project}/{run_id}")
        except wandb.errors.CommError:
            print(f"Run {run_id} not found in project {project}")
            return None
        
        # Load from artifacts
        try:
            # Get all logged artifacts from this run
            artifacts = list(run.logged_artifacts())
            explanations_artifacts = [a for a in artifacts if a.type == "explanations" and "evaluation_explanations" in a.name]
            
            if not explanations_artifacts:
                print(f"No explanations found in run {run_id}")
                return None
            
            # Select artifact based on version parameter
            if version is None:
                # Use the latest explanations artifact (highest version)
                selected_artifact = max(explanations_artifacts, key=lambda x: x.version)
                print(f"Using latest version (v{selected_artifact.version})")
            else:
                # Find artifact with specific version
                if isinstance(version, int):
                    version_str = f"v{version}"
                elif version == "latest":
                    selected_artifact = max(explanations_artifacts, key=lambda x: x.version)
                    print(f"Using latest version (v{selected_artifact.version})")
                else:
                    version_str = version if version.startswith("v") else f"v{version}"
                
                if version != "latest":
                    matching_artifacts = [a for a in explanations_artifacts if a.version == version_str]
                    if not matching_artifacts:
                        available_versions = [a.version for a in explanations_artifacts]
                        print(f"Version {version_str} not found. Available versions: {available_versions}")
                        return None
                    selected_artifact = matching_artifacts[0]
                    print(f"Using specified version {version_str}")
            
            # Download to the specified output path
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            download_dir = output_path / f"downloaded_explanations_{run_id}"
            
            artifact_dir = selected_artifact.download(root=str(download_dir))
            
            explanations_path = Path(artifact_dir) / "explanations.json"
            if explanations_path.exists():
                with open(explanations_path, "r") as f:
                    explanations = json.load(f)
                print(f"Loaded explanations from Wandb artifact: {selected_artifact.name} ({selected_artifact.version})")
                
                # Also load summary if available
                summary_path = Path(artifact_dir) / "explanation_summary.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    print(f"  Total explanations: {summary.get('num_explanations', 'N/A')}")
                    for layer, count in summary.get('explained_neurons_per_layer', {}).items():
                        print(f"  {layer}: {count} neurons")
                
                # Clean up downloaded files after loading
                try:
                    import shutil
                    shutil.rmtree(download_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up downloaded files at {download_dir}: {e}")
                
                return explanations
        except Exception as e:
            print(f"Could not load explanations from artifacts: {e}")
            return None
        
        print(f"No explanations found in run {run_id}")
        return None
    except Exception as e:
        print(f"Error loading explanations from Wandb: {e}")
        return None
