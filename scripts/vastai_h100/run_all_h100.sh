#!/bin/bash
# Script to run all H100 training scripts in separate tmux sessions

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# List of scripts to run
SCRIPTS=(
    "baselines.sh"
    "run_lagrangian_0.sh"
    "run_lagrangian_1.sh"
    "run_lagrangian_2.sh"
    "run_matryoshka_baseline.sh"
    "run_matryoshka_lagrangian_0.sh"
    "run_matryoshka_lagrangian_1.sh"
    "run_matryoshka_lagrangian_2.sh"
)

echo "Starting all H100 training scripts in separate tmux sessions..."
echo "Project root: $PROJECT_ROOT"
echo ""

for script in "${SCRIPTS[@]}"; do
    # Create session name from script name (remove .sh extension)
    session_name="${script%.sh}"
    script_path="$SCRIPT_DIR/$script"
    
    # Check if script exists
    if [[ ! -f "$script_path" ]]; then
        echo "Warning: Script not found: $script_path"
        continue
    fi
    
    # Check if tmux session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session '$session_name' already exists, skipping..."
        continue
    fi
    
    echo "Starting tmux session: $session_name"
    
    # Create new tmux session and run the script
    tmux new-session -d -s "$session_name" -c "$PROJECT_ROOT" "bash $script_path; exec bash"
done

echo ""
echo "All sessions started. Use 'tmux ls' to list sessions."
echo "Attach to a session with: tmux attach -t <session_name>"
echo ""
echo "Available sessions:"
tmux ls 2>/dev/null || echo "No tmux sessions running"
