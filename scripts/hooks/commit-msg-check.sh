#!/bin/sh

# Regex to match the commit message structure
COMMIT_MSG_REGEX="^(feat|fix|data|experiment|model|docs|refactor|test|chore|build|ci|revert|style)(\([^()]+\))?: .+ \[#([A-Z]+-[0-9]+)\]$"

# Get the commit message
COMMIT_MSG_FILE="$1"
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check if the commit message matches the expected structure
if ! echo "$COMMIT_MSG" | grep -qE "$COMMIT_MSG_REGEX"; then
  echo "Error: Commit message must follow the format '<type>(<scope>): <subject> [#JIRA-123]'"
  echo "Example: feat(auth): add user authentication feature [#DATA-123]"
  exit 1
fi
