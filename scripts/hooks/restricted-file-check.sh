#!/bin/bash

# List of files that should not be modified
restricted_files=(
  ".vscode/settings.json"
  ".vscode/extensions.json"
  ".github/black.yaml"
  ".github/ruff.yaml"
  "Dockerfile"
  "site/"
)

# Function to check if a file is in the restricted list
is_restricted_file() {
  for file in "${restricted_files[@]}"; do
    if [ "$1" = "$file" ]; then
      return 0
    fi
  done
  return 1
}

# Get the list of modified files in the commit
modified_files=$(git diff --cached --name-only)

# Check each modified file
for file in $modified_files; do
  if is_restricted_file "$file"; then
    echo "Error: Modifications to '$file' are not allowed."
    exit 1
  fi
done
