#!/usr/bin/env python3
"""
Script to copy a W&B project from one entity to another.

This script copies all runs, including their configs, metrics history, 
summary metrics, and artifacts from a source project to a destination project.

Usage:
    python scripts/copy_wandb_project.py
    
Or with custom arguments:
    python scripts/copy_wandb_project.py --src-entity raymondl --dst-entity lagrangian-sae --project gpt2-small
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
from tqdm import tqdm

from settings import settings


def copy_wandb_project(
    src_entity: str,
    dst_entity: str,
    project_name: str,
    copy_artifacts: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Copy all runs from a source W&B project to a destination project.
    
    Args:
        src_entity: Source entity (username or team name)
        dst_entity: Destination entity (username or team name)
        project_name: Name of the project to copy
        copy_artifacts: Whether to copy artifacts (model files, etc.)
        dry_run: If True, only print what would be done without actually copying
    """
    # Login to wandb
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    else:
        wandb.login()
    
    api = wandb.Api()
    
    src_project_path = f"{src_entity}/{project_name}"
    dst_project_path = f"{dst_entity}/{project_name}"
    
    print(f"Copying project: {src_project_path} -> {dst_project_path}")
    print(f"Copy artifacts: {copy_artifacts}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)
    
    # Get all runs from source project
    try:
        runs = api.runs(src_project_path)
        runs_list = list(runs)  # Convert to list to get count
        print(f"Found {len(runs_list)} runs in source project")
    except Exception as e:
        print(f"Error fetching runs from {src_project_path}: {e}")
        return
    
    if dry_run:
        print("\n[DRY RUN] Would copy the following runs:")
        for run in runs_list:
            print(f"  - {run.name} (id: {run.id}, state: {run.state})")
        return
    
    # Copy each run
    for src_run in tqdm(runs_list, desc="Copying runs"):
        print(f"\nCopying run: {src_run.name} (id: {src_run.id})")
        
        try:
            # Get run details
            config = dict(src_run.config)
            summary = dict(src_run.summary)
            tags = src_run.tags
            notes = src_run.notes
            
            # Remove internal wandb keys from summary
            summary_clean = {k: v for k, v in summary.items() if not k.startswith("_")}
            
            # Initialize new run in destination
            new_run = wandb.init(
                entity=dst_entity,
                project=project_name,
                name=src_run.name,
                config=config,
                tags=tags,
                notes=notes,
                reinit=True,
            )
            
            # Log the summary metrics
            if summary_clean:
                wandb.log(summary_clean)
            
            # Try to copy full history (metrics over time)
            try:
                history = src_run.history(samples=10000, pandas=False)
                for step_data in history:
                    # Filter out internal wandb keys
                    step_clean = {k: v for k, v in step_data.items() 
                                  if not k.startswith("_") and k != "step"}
                    if step_clean:
                        wandb.log(step_clean)
            except Exception as e:
                print(f"  Warning: Could not copy full history: {e}")
            
            # Copy artifacts if requested
            if copy_artifacts:
                try:
                    # Reserved artifact types that cannot be copied
                    RESERVED_TYPES = {
                        "wandb-history",
                        "wandb-metadata", 
                        "wandb-summary",
                        "wandb-config",
                        "code",
                        "run_table",
                    }
                    
                    artifacts = src_run.logged_artifacts()
                    for artifact in artifacts:
                        # Skip reserved internal artifact types
                        if artifact.type in RESERVED_TYPES:
                            print(f"  Skipping internal artifact: {artifact.name} (type: {artifact.type})")
                            continue
                            
                        print(f"  Copying artifact: {artifact.name} (type: {artifact.type})")
                        # Download artifact to temp directory
                        artifact_dir = artifact.download()
                        # Log to new run
                        new_artifact = wandb.Artifact(
                            name=artifact.name.split(":")[0],  # Remove version
                            type=artifact.type,
                            description=artifact.description,
                            metadata=artifact.metadata,
                        )
                        new_artifact.add_dir(artifact_dir)
                        new_run.log_artifact(new_artifact)
                except Exception as e:
                    print(f"  Warning: Could not copy artifacts: {e}")
            
            wandb.finish()
            print(f"  ✓ Successfully copied run: {src_run.name}")
            
        except Exception as e:
            print(f"  ✗ Error copying run {src_run.name}: {e}")
            try:
                wandb.finish()
            except:
                pass
    
    print("\n" + "=" * 60)
    print(f"Project copy complete: {src_project_path} -> {dst_project_path}")
    print(f"View at: https://wandb.ai/{dst_project_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy a W&B project from one entity to another"
    )
    parser.add_argument(
        "--src-entity",
        type=str,
        default="raymondl",
        help="Source entity (username or team name)",
    )
    parser.add_argument(
        "--dst-entity",
        type=str,
        default="lagrangian-sae",
        help="Destination entity (username or team name)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="gpt2-small",
        help="Project name to copy",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip copying artifacts (faster, but no model files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't actually copy",
    )
    
    args = parser.parse_args()
    
    copy_wandb_project(
        src_entity=args.src_entity,
        dst_entity=args.dst_entity,
        project_name=args.project,
        copy_artifacts=not args.no_artifacts,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

