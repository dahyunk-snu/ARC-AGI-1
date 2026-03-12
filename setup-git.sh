#!/bin/bash

# Set git commit template
git config commit.template "./git-commit-template.md"

# Set git to push to the current branch by default
git config push.default current

echo "Git configuration complete:"
echo "- Commit template: ./git-commit-template.md"
echo "- push.default: current"
