#!/bin/bash

# Regex pattern for snake_case filenames
SNAKE_CASE_REGEX='^[a-z0-9_]+(\.py|\.ipynb|\.md|\.csv|\.json|\.jsonl)$'

# List of allowed filenames that do not follow snake_case
ALLOWED_FILENAMES=("README.md" "CONTRIBUTING.md" "__init__.py")

# List of directories to inspect
DIRECTORIES_TO_INSPECT=("src/" "tests/" "notebooks/")

# Get the list of files to be committed
files_to_check=$(git diff --cached --name-only)

# Function to check if a filename is in the allowed list
is_allowed_filename() {
  local filename=$1
  for allowed in "${ALLOWED_FILENAMES[@]}"; do
    if [[ $filename == $allowed ]]; then
      return 0
    fi
  done
  return 1
}

# Function to check if a file is in the directories to inspect
is_in_inspected_directory() {
  local file=$1
  for directory in "${DIRECTORIES_TO_INSPECT[@]}"; do
    if [[ $file == $directory* ]]; then
      return 0
    fi
  done
  return 1
}

# Check each file
for file in $files_to_check; do
  if is_in_inspected_directory "$file" && [[ $file =~ \.py$|\.ipynb$|\.md$|\.csv$|\.json$|\.jsonl$ ]]; then
    filename=$(basename "$file")
    if ! is_allowed_filename "$filename" && ! [[ $filename =~ $SNAKE_CASE_REGEX ]]; then
      echo "Error: Filename '$filename' in '$file' does not follow the snake_case naming convention."
      exit 1
    fi
  fi
done

# If no issues, exit with status 0
exit 0
