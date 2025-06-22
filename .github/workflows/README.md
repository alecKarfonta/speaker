# Speaker TTS API CI/CD Workflows

This directory contains GitHub Actions workflows for the Speaker TTS API project.

## Workflows

### 1. `package.yml` - Build and Publish Packages
**Trigger:** Push to `main` branch or manual dispatch

**Purpose:** Builds and publishes Docker images to GitHub Container Registry (GHCR)

**Features:**
- Builds Docker image with version tagging
- Publishes to GHCR with multiple tags (version, latest, commit SHA)
- Creates GitHub releases automatically
- Includes comprehensive logging and validation

**Output:**
- Docker images: `ghcr.io/aleckarfonta/speaker:v1.0.0`, `ghcr.io/aleckarfonta/speaker:latest`
- GitHub release with release notes

### 2. `deploy.yml` - TTS API CI/CD Pipeline
**Trigger:** Push to `dev` branch, PR to `main`, or manual dispatch

**Purpose:** Complete CI/CD pipeline with testing, building, and deployment

**Jobs:**

#### Quality Check
- Python dependency installation
- Code formatting with `ruff`
- Linting with `ruff`
- Security scanning with `bandit`
- Test execution with coverage reporting
- Codecov integration

#### Frontend Quality
- Node.js setup and dependency installation
- Frontend build verification
- Artifact upload

#### Build and Push
- Docker image building and pushing
- Multi-platform support (linux/amd64)
- GHCR integration

#### Security Scan
- Trivy vulnerability scanning
- SARIF report upload to GitHub Security tab

#### Deploy to Development
- Kubernetes deployment to dev environment
- Smoke tests
- Health checks

#### Deploy to Production
- Kubernetes deployment to production environment
- Production health checks
- Slack notifications

## Required Secrets

### GitHub Secrets
- `GHCR_TOKEN`: GitHub Container Registry token (optional, falls back to `GITHUB_TOKEN`)
- `GITHUB_TOKEN`: Default GitHub token (automatically provided)

### Kubernetes Secrets
- `KUBECONFIG_DEV`: Base64-encoded kubeconfig for development cluster
- `KUBECONFIG_PROD`: Base64-encoded kubeconfig for production cluster

### Optional Secrets
- `SLACK_WEBHOOK`: Slack webhook URL for deployment notifications

## Version Management

The project uses `app/version.py` for version management. The CI/CD pipeline automatically extracts the version from this file for Docker image tagging and GitHub releases.

## Kubernetes Deployment

The `k8s/` directory contains Kubernetes manifests for:
- Namespace configuration
- ConfigMap for environment variables
- Secrets template
- PersistentVolumeClaim for model cache
- Deployment with GPU support
- Service and Ingress
- HorizontalPodAutoscaler
- ServiceMonitor for Prometheus

## Usage

### Development Workflow
1. Create feature branch from `dev`
2. Make changes and test locally
3. Push to feature branch
4. Create PR to `dev`
5. CI/CD pipeline runs automatically
6. Deploy to development environment on merge

### Production Release
1. Create PR from `dev` to `main`
2. CI/CD pipeline runs tests and builds
3. Merge to `main` triggers production deployment
4. Docker image published to GHCR
5. GitHub release created automatically

## Monitoring

The deployment includes:
- Health checks and readiness probes
- Prometheus ServiceMonitor
- Resource monitoring and autoscaling
- Slack notifications for deployments 