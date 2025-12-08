#!/bin/bash
# Pre-push safety check script
# Run this before pushing to GitHub to verify no sensitive data or large files

echo "==================================================================="
echo "GITHUB PUSH SAFETY CHECK"
echo "==================================================================="

echo ""
echo "Checking for large files (>100MB)..."
large_files=$(find . -type f -size +100M 2>/dev/null | grep -v ".git" | grep -v "venv")
if [ -z "$large_files" ]; then
    echo "✓ No large files found"
else
    echo "✗ WARNING: Large files detected (these should be in .gitignore):"
    echo "$large_files"
    echo ""
    echo "If these appear in 'git status', update .gitignore!"
fi

echo ""
echo "Checking for .env file..."
if [ -f .env ]; then
    if grep -q "^\.env$" .gitignore 2>/dev/null; then
        echo "✓ .env exists and is in .gitignore"
    else
        echo "✗ WARNING: .env exists but NOT in .gitignore!"
    fi
else
    echo "✓ No .env file"
fi

echo ""
echo "Checking git status..."
if [ -d .git ]; then
    echo "Git repository initialized"
    echo ""
    echo "Files to be committed:"
    git status --short | head -20
    
    echo ""
    echo "Checking for large files in staging area..."
    large_staged=$(git ls-files -s | awk '{if ($4 > 100000000) print $4}' | wc -l)
    if [ "$large_staged" -eq 0 ]; then
        echo "✓ No large files in staging area"
    else
        echo "✗ WARNING: Large files in staging area!"
    fi
else
    echo "Git not initialized yet. Run: git init"
fi

echo ""
echo "==================================================================="
echo "SAFE TO PUSH IF:"
echo "==================================================================="
echo "1. ✓ No .env or secrets in 'git status'"
echo "2. ✓ No data/models/ files in 'git status'"
echo "3. ✓ No venv/ files in 'git status'"
echo "4. ✓ No large (>100MB) files in 'git status'"
echo "5. ✓ Only code, docs, and configs appear"
echo ""
echo "Run 'git status' to verify before pushing!"
echo "==================================================================="
