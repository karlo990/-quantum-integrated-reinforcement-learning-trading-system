#!/bin/bash
# ============================================================================
# K1RL QUASAR — HuggingFace Spaces Deployment Script
# ============================================================================

set -e

echo "🌌 K1RL QUASAR — HuggingFace Spaces Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HF_USERNAME="KarlQuant"
SPACE_NAME="k1rl-quasar"
SPACE_REPO="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo -e "${BLUE}Configuration:${NC}"
echo "  HF Username: ${HF_USERNAME}"
echo "  Space Name: ${SPACE_NAME}"
echo "  Repository: ${SPACE_REPO}"
echo ""

# Step 1: Clone the Space repository
echo -e "${YELLOW}Step 1: Cloning HuggingFace Space repository...${NC}"
if [ -d "${SPACE_NAME}" ]; then
    echo -e "${YELLOW}Directory exists, updating...${NC}"
    cd "${SPACE_NAME}"
    git pull
else
    git clone "${SPACE_REPO}"
    cd "${SPACE_NAME}"
fi
echo -e "${GREEN}✅ Repository ready${NC}"
echo ""

# Step 2: Copy deployment files
echo -e "${YELLOW}Step 2: Copying deployment files  ...${NC}"

# Core configuration files
cp ../Dockerfile .
cp ../supervisord.conf .
cp ../redis_hf.conf .
cp ../requirements.txt .
cp ../README.md .

# Python services
cp ../dashboard_service.py .
cp ../quasar_service.py .
cp ../health_monitor.py .
cp ../sleep_prevention.py .
cp ../redis_config.py .
cp ../visitor_tracker.py .
cp ../app.py .

echo -e "${GREEN}✅ Deployment files copied${NC}"
echo ""

# Step 3: Create HuggingFace Spaces metadata
echo -e "${YELLOW}Step 3: Creating HuggingFace Spaces configuration...${NC}"
cat > README.md << 'EOF'
---
title: K1RL QUASAR — Quantitative Intelligence Observatory
emoji: 🌌
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI Training Monitoring & Analytics Platform with Real-time Metrics
tags:
  - machine-learning
  - monitoring
  - analytics
  - training
  - dashboard
  - ai
  - pytorch
  - redis
  - flask
models:
  - KarlQuant/k1rl-checkpoints
datasets:
  - KarlQuant/k1rl-checkpoints
---

# K1RL QUASAR — Quantitative Intelligence Observatory

> **Production AI Training Monitoring Platform** | Real-time Metrics | 24/7 Uptime

## 🌌 Live Dashboard

This HuggingFace Space runs a comprehensive AI training monitoring platform featuring:

- **Real-time Training Metrics** — Loss curves, accuracy, buffer statistics
- **System Health Monitoring** — RAM/CPU usage with auto-cleanup
- **Visitor Analytics** — Global visitor tracking and statistics  
- **Service Management** — Multi-process architecture with Redis
- **Sleep Prevention** — Automatic uptime maintenance
- **Checkpoint Management** — HuggingFace Hub integration

## 🚀 Services

The platform runs multiple coordinated services:

1. **Dashboard** (Port 7860) — Web interface and API
2. **Quasar Engine** — Main AI training loop
3. **Health Monitor** — Resource management and cleanup
4. **Redis Server** — Real-time metrics storage
5. **Sleep Prevention** — 24/7 uptime maintenance

## 📊 API Endpoints

- `/` — Main dashboard interface
- `/health` — Health check and uptime
- `/api/metrics` — Training metrics JSON
- `/api/visitors` — Visitor analytics
- `/logs/{service}` — Service logs

## 🔧 Architecture

Built for HuggingFace Spaces 12GB environment with:
- Docker containerization
- Supervisor process management
- Automatic resource cleanup
- Graceful error recovery
- Cloud checkpoint backup

---

*Powered by K1RL Quantitative Intelligence Research*
EOF

echo -e "${GREEN}✅ HuggingFace configuration created${NC}"
echo ""

# Step 4: Check for required original files
echo -e "${YELLOW}Step 4: Checking for original project files...${NC}"

REQUIRED_FILES=(
    "quasar_main4.py"
    "Features.py" 
    "Rewards.py"
    "dashboard.html"
    "quasar_plasticity_module.py"
)

MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "../${file}" ]; then
        MISSING_FILES+=("${file}")
    else
        cp "../${file}" .
        echo -e "${GREEN}✓${NC} ${file}"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}  - ${file}${NC}"
    done
    echo ""
    echo -e "${YELLOW}Please copy these files manually before deployment.${NC}"
else
    echo -e "${GREEN}✅ All required files present${NC}"
fi
echo ""

# Step 5: Git setup and commit
echo -e "${YELLOW}Step 5: Preparing Git commit...${NC}"

git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${BLUE}Changes detected, creating commit...${NC}"
    git config --global user.email "action@github.com" 2>/dev/null || true
    git config --global user.name "K1RL Deployment" 2>/dev/null || true
    
    git commit -m "K1RL QUASAR - Production deployment to HuggingFace Spaces

🌌 Features:
- Real-time training monitoring dashboard
- Multi-service architecture with Redis
- Health monitoring and automatic cleanup  
- Visitor analytics and geolocation
- Sleep prevention for 24/7 uptime
- HuggingFace Hub checkpoint management

🚀 Services:
- Dashboard (Port 7860)
- Quasar Engine (Training)
- Health Monitor (Resource management)
- Redis Server (Metrics storage)
- Sleep Prevention (Uptime)

🐳 Deployment: Docker + Supervisor + 12GB RAM optimization"

    echo -e "${GREEN}✅ Commit created${NC}"
fi
echo ""

# Step 6: Display final instructions
echo -e "${BLUE}=============================================="
echo -e "🎉 Deployment preparation complete!"
echo -e "===============================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo -e "${BLUE}1. Configure HuggingFace Spaces Secrets:${NC}"
echo "   Go to: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}/settings"
echo ""
echo "   Add these secrets:"
echo -e "${GREEN}   Name: HF_TOKEN${NC}"
echo "   Value: YOUR_HF_TOKEN_HERE"
echo ""
echo -e "${GREEN}   Name: HF_USERNAME${NC}"
echo "   Value: ${HF_USERNAME}"
echo ""
echo -e "${GREEN}   Name: REDIS_PASSWORD${NC}"
echo "   Value: k1rl_099a0c008e32300dc3c14189"
echo ""

echo -e "${BLUE}2. Deploy to HuggingFace Spaces:${NC}"
echo "   git push"
echo ""

echo -e "${BLUE}3. Monitor deployment:${NC}"
echo "   Watch the build logs in HuggingFace Spaces"
echo "   Access your dashboard at: https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo ""

echo -e "${BLUE}4. Verify services:${NC}"
echo "   Health check: https://${HF_USERNAME}-${SPACE_NAME}.hf.space/health"
echo "   Metrics API: https://${HF_USERNAME}-${SPACE_NAME}.hf.space/api/metrics"
echo "   Service logs: https://${HF_USERNAME}-${SPACE_NAME}.hf.space/logs/dashboard"
echo ""

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo -e "${RED}⚠️  WARNING: Missing files detected${NC}"
    echo -e "${YELLOW}   Copy these files before pushing:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}   - ${file}${NC}"
    done
    echo ""
fi

echo -e "${GREEN}🚀 Ready for deployment!${NC}"
echo ""
echo -e "${YELLOW}Current directory: $(pwd)${NC}"
echo -e "${YELLOW}Run 'git push' to deploy to HuggingFace Spaces${NC}"
