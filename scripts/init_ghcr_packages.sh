#!/bin/bash
set -e

# Script to initialize GitHub Container Registry packages
# This creates the packages with proper permissions before CI/CD pushes to them

REGISTRY="ghcr.io"
GITHUB_USER="${GITHUB_REPOSITORY_OWNER:-aleckarfonta}"
BACKEND_IMAGE="${REGISTRY}/${GITHUB_USER}/speaker"
FRONTEND_IMAGE="${REGISTRY}/${GITHUB_USER}/speaker-frontend"

echo "Initializing GHCR packages for ${GITHUB_USER}..."
echo ""
echo "1. Make sure you're logged into GHCR:"
echo "   echo \$GITHUB_TOKEN | docker login ghcr.io -u ${GITHUB_USER} --password-stdin"
echo ""
echo "   If you don't have a token, create one at:"
echo "   https://github.com/settings/tokens/new"
echo "   Scopes needed: write:packages, read:packages, delete:packages"
echo ""

read -p "Are you logged into GHCR? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please login first and re-run this script"
    exit 1
fi

echo ""
echo "Building and pushing initial backend image..."
docker build -t "${BACKEND_IMAGE}:init" -f Dockerfile .
docker push "${BACKEND_IMAGE}:init"

echo ""
echo "Building and pushing initial frontend image..."
docker build -t "${FRONTEND_IMAGE}:init" -f frontend/Dockerfile frontend/
docker push "${FRONTEND_IMAGE}:init"

echo ""
echo "Initial packages created! Now you need to:"
echo "1. Go to https://github.com/${GITHUB_USER}?tab=packages"
echo "2. For each package (speaker and speaker-frontend):"
echo "   a. Click 'Package settings'"
echo "   b. Scroll to 'Manage Actions access'"
echo "   c. Add your repository with 'Write' role"
echo "   d. Under 'Danger Zone', change visibility if needed"
echo ""
echo "After that, your GitHub Actions workflow should work!"
