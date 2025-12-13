#!/usr/bin/env python3
"""
Script to copy W&B artifacts from one project to another.

Use this AFTER moving the project/runs via the W&B UI, since artifacts
don't move with runs. This only copies the artifacts, not metrics/history.

Usage:
    python scripts/copy_wandb_artifacts.py
    
Or with custom arguments:
    python scripts/copy_wandb_artifacts.py --src-entity raymondl --dst-entity lagrangian-sae --project gpt2-small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
from tqdm import tqdm

from settings import settings

# Reserved artifact types that are internal to wandb (skip these)
RESERVED_TYPES = {
    "wandb-history",
    "wandb-metadata",
    "wandb-summary",
    "wandb-config",
    "code",
    "run_table",
    "job",
}


def copy_artifacts(
    src_entity: str,
    dst_entity: str,
    project_name: str,
    dry_run: bool = False,
) -> None:
    """
    Copy all artifacts from source project to destination project.
    
    Args:
        src_entity: Source entity (username or team name)
        dst_entity: Destination entity (username or team name)  
        project_name: Name of the project
        dry_run: If True, only print what would be done
    """
    if settings.wandb_api_key:
        wandb.login(key=settings.wandb_api_key)
    else:
        wandb.login()
    
    api = wandb.Api()
    
    src_project_path = f"{src_entity}/{project_name}"
    dst_project_path = f"{dst_entity}/{project_name}"
    
    print(f"Copying artifacts: {src_project_path} -> {dst_project_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)
    
    # Get all artifact types in the project
    try:
        artifact_types = api.artifact_types(project=src_project_path)
        artifact_types_list = list(artifact_types)
        print(f"Found {len(artifact_types_list)} artifact types")
    except Exception as e:
        print(f"Error fetching artifact types: {e}")
        return
    
    # Filter out reserved types
    valid_types = [at for at in artifact_types_list if at.name not in RESERVED_TYPES]
    skipped_types = [at.name for at in artifact_types_list if at.name in RESERVED_TYPES]
    
    if skipped_types:
        print(f"Skipping reserved types: {skipped_types}")
    
    if not valid_types:
        print("No user artifacts to copy!")
        return
    
    print(f"Will copy {len(valid_types)} artifact types: {[at.name for at in valid_types]}")
    
    if dry_run:
        print("\n[DRY RUN] Would copy:")
        for artifact_type in valid_types:
            collections = list(artifact_type.collections())
            for collection in collections:
                versions = list(collection.versions())
                for v in versions:
                    print(f"  - {v.name} (type: {artifact_type.name}, size: {v.size})")
        return
    
    # Initialize a run in destination to log artifacts
    dst_run = wandb.init(
        entity=dst_entity,
        project=project_name,
        name="artifact-migration",
        job_type="migration",
    )
    
    total_copied = 0
    
    for artifact_type in valid_types:
        print(f"\nProcessing artifact type: {artifact_type.name}")
        
        try:
            collections = list(artifact_type.collections())
            
            for collection in tqdm(collections, desc=f"  {artifact_type.name}"):
                versions = list(collection.versions())
                
                for artifact in versions:
                    try:
                        print(f"    Copying: {artifact.name}")
                        
                        # Download artifact
                        artifact_dir = artifact.download()
                        
                        # Create new artifact with same name (without version suffix)
                        base_name = artifact.name.split(":")[0]
                        new_artifact = wandb.Artifact(
                            name=base_name,
                            type=artifact_type.name,
                            description=artifact.description or "",
                            metadata=dict(artifact.metadata) if artifact.metadata else {},
                        )
                        new_artifact.add_dir(artifact_dir)
                        dst_run.log_artifact(new_artifact)
                        total_copied += 1
                        
                    except Exception as e:
                        print(f"    Error copying {artifact.name}: {e}")
                        
        except Exception as e:
            print(f"  Error processing type {artifact_type.name}: {e}")
    
    wandb.finish()
    
    print("\n" + "=" * 60)
    print(f"Copied {total_copied} artifacts to {dst_project_path}")
    print(f"View at: https://wandb.ai/{dst_project_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy W&B artifacts from one project to another"
    )
    parser.add_argument(
        "--src-entity",
        type=str,
        default="raymondl",
        help="Source entity",
    )
    parser.add_argument(
        "--dst-entity",
        type=str,
        default="lagrangian-sae",
        help="Destination entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="gpt2-small",
        help="Project name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    
    args = parser.parse_args()
    
    copy_artifacts(
        src_entity=args.src_entity,
        dst_entity=args.dst_entity,
        project_name=args.project,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

