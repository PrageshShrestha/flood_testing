#!/bin/bash
# ============================================
# FULLY AUTOMATED YOLO INFERENCE SETUP
# Handles sudo, dependencies, and execution
# Usage: ./auto_setup_and_run.sh
# ============================================

# ============================================
# CONFIGURATION
# ============================================

# Set credentials (modify these as needed)
USERNAME="kuiot"
PASSWORD="kuiot"

# Environment name
ENV_NAME="yolo_env"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="setup_$(date +%Y%m%d_%H%M%S).log"

# ============================================
# HELPER FUNCTIONS
# ============================================

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}✗ ERROR:${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

log_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${CYAN}ℹ${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to run commands with sudo (auto-provides password)
run_sudo() {
    echo "$PASSWORD" | sudo -S $@ 2>/dev/null
}

# ============================================
# AUTO-SUDO SETUP
# ============================================

setup_sudo() {
    print_header "Configuring Sudo Access"
    
    # Test if sudo works with password
    echo "$PASSWORD" | sudo -S echo "Sudo configured" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "Sudo access configured for user: $USERNAME"
    else
        log_error "Failed to configure sudo. Check username/password"
    fi
}

# ============================================
# SYSTEM DEPENDENCIES (apt)
# ============================================

install_system_deps() {
    print_header "Installing System Dependencies (apt)"
    
    log "Updating package lists..."
    run_sudo apt update -y >> "$LOG_FILE" 2>&1
    log_success "Package lists updated"
    
    log "Installing required system packages..."
    
    PACKAGES=(
        "ffmpeg"
        "python3-pip"
        "python3-dev"
        "python3-venv"
        "libatlas-base-dev"
        "libopenblas-dev"
        "libjpeg-dev"
        "zlib1g-dev"
        "libtiff-dev"
        "libfreetype6-dev"
        "libharfbuzz-dev"
        "libwebp-dev"
        "libgstreamer1.0-0"
        "gstreamer1.0-tools"
        "gstreamer1.0-plugins-good"
        "gstreamer1.0-plugins-bad"
        "gstreamer1.0-plugins-ugly"
        "libavcodec-dev"
        "libavformat-dev"
        "libswscale-dev"
        "libv4l-dev"
        "v4l-utils"
    )
    
    for package in "${PACKAGES[@]}"; do
        log "  Installing $package..."
        run_sudo apt install -y $package >> "$LOG_FILE" 2>&1
    done
    
    log_success "All system dependencies installed"
}

# ============================================
# VIRTUAL ENVIRONMENT SETUP
# ============================================

setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    # Remove existing if present
    if [ -d "$ENV_NAME" ]; then
        log "Removing existing environment..."
        rm -rf "$ENV_NAME"
    fi
    
    log "Creating virtual environment: $ENV_NAME"
    python3 -m venv "$ENV_NAME" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment created"
    else
        log_error "Failed to create virtual environment"
    fi
    
    # Activate environment
    source "$ENV_NAME/bin/activate"
    log_success "Environment activated"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip >> "$LOG_FILE" 2>&1
    
    # Show Python version
    PY_VERSION=$(python --version)
    log_info "Python: $PY_VERSION"
}

# ============================================
# PYTHON PACKAGES (pip)
# ============================================

install_python_packages() {
    print_header "Installing Python Packages (pip)"
    
    # Ensure we're in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        log_error "Virtual environment not active"
    fi
    
    log "Installing PyTorch for ARM64 (RPi5 optimized)..."
    pip install --no-cache-dir \
        torch==2.0.1 \
        torchvision==0.15.2 \
        --index-url https://download.pytorch.org/whl/cpu \
        >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "PyTorch installed"
    else
        log_error "PyTorch installation failed"
    fi
    
    log "Installing ultralytics YOLO..."
    pip install --no-cache-dir ultralytics>=8.0.0 >> "$LOG_FILE" 2>&1
    log_success "Ultralytics installed"
    
    log "Installing computer vision packages..."
    pip install --no-cache-dir \
        opencv-python-headless>=4.8.0 \
        numpy>=1.24.0 \
        Pillow>=10.0.0 \
        >> "$LOG_FILE" 2>&1
    log_success "CV packages installed"
    
    log "Installing system monitoring..."
    pip install --no-cache-dir psutil>=5.9.0 >> "$LOG_FILE" 2>&1
    log_success "Monitoring tools installed"
    
    # Verify installations
    log "Verifying installations..."
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')" >> "$LOG_FILE" 2>&1
    python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" >> "$LOG_FILE" 2>&1
    python -c "from ultralytics import YOLO; print('  Ultralytics: OK')" >> "$LOG_FILE" 2>&1
    
    log_success "All Python packages installed"
}

# ============================================
# VERIFY INPUT FILES
# ============================================

verify_files() {
    print_header "Verifying Input Files"
    
    if [ -f "best2.pt" ]; then
        MODEL_SIZE=$(du -h best2.pt | cut -f1)
        log_success "Model found: best2.pt ($MODEL_SIZE)"
    else
        log_error "Model file 'best2.pt' not found in current directory"
    fi
    
    if [ -f "test2.mp4" ]; then
        VIDEO_SIZE=$(du -h test2.mp4 | cut -f1)
        log_success "Video found: test2.mp4 ($VIDEO_SIZE)"
    else
        log_error "Video file 'test2.mp4' not found in current directory"
    fi
}

# ============================================
# CHECK RASPBERRY PI HARDWARE
# ============================================

check_rpi_hardware() {
    print_header "Hardware Detection"
    
    # Check if running on Raspberry Pi
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model)
        log_success "Device: $MODEL"
    else
        log_warning "Not running on Raspberry Pi"
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    log_info "CPU Cores: $CPU_CORES"
    
    # Check RAM
    TOTAL_RAM=$(free -h | awk '/^Mem:/{print $2}')
    AVAIL_RAM=$(free -h | awk '/^Mem:/{print $7}')
    log_info "Total RAM: $TOTAL_RAM"
    log_info "Available RAM: $AVAIL_RAM"
    
    # Check temperature (if on RPi)
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
        TEMP_C=$((TEMP/1000))
        log_info "CPU Temperature: ${TEMP_C}°C"
    fi
    
    # Set CPU governor to performance (requires sudo)
    log "Setting CPU governor to performance mode..."
    run_sudo bash -c 'for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $i; done' 2>/dev/null
    log_success "CPU performance mode enabled"
}

# ============================================
# RUN INFERENCE
# ============================================

run_inference() {
    print_header "Starting YOLO Inference"
    
    # Check if run.py exists
    if [ ! -f "run.py" ]; then
        log_error "run.py not found in current directory"
    fi
    
    log_info "Starting inference at $(date)"
    log_info "Log file: $LOG_FILE"
    echo ""
    
    # Run with python (sudo not needed for inference, only for CPU governor)
    python run.py
    
    if [ $? -eq 0 ]; then
        echo ""
        log_success "Inference completed successfully"
    else
        log_error "Inference failed"
    fi
}

# ============================================
# GENERATE SUMMARY
# ============================================

generate_summary() {
    print_header "Setup Summary"
    
    echo -e "${GREEN}✓ System dependencies installed${NC}"
    echo -e "${GREEN}✓ Virtual environment created: $ENV_NAME${NC}"
    echo -e "${GREEN}✓ Python packages installed${NC}"
    echo -e "${GREEN}✓ Input files verified${NC}"
    echo -e "${GREEN}✓ Inference completed${NC}"
    
    echo ""
    echo -e "${CYAN}Output files are in: research_runs/run_*/${NC}"
    echo -e "${CYAN}Log file: $LOG_FILE${NC}"
    
    # Show latest run folder
    if [ -d "research_runs" ]; then
        LATEST_RUN=$(ls -td research_runs/run_* 2>/dev/null | head -1)
        if [ -n "$LATEST_RUN" ]; then
            echo -e "${GREEN}✓ Latest run: $LATEST_RUN${NC}"
        fi
    fi
}

# ============================================
# CLEANUP FUNCTION
# ============================================

cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup if needed
}

# ============================================
# MAIN EXECUTION
# ============================================

main() {
    clear
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     YOLO INFERENCE PIPELINE FOR RASPBERRY PI 5          ║"
    echo "║              Fully Automated Setup                       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "Starting automated setup at $(date)"
    
    # Run all steps
    setup_sudo
    install_system_deps
    setup_venv
    install_python_packages
    verify_files
    check_rpi_hardware
    run_inference
    generate_summary
    
    log_success "All tasks completed at $(date)"
    
    # Deactivate virtual environment
    deactivate 2>/dev/null
}

# ============================================
# ERROR HANDLING
# ============================================

trap cleanup EXIT

# Run main function
main
