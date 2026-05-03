#!/bin/bash
# ============================================
# YOLO INFERENCE PACKAGE INSTALLER
# Install dependencies only (runs inside existing venv)
# Usage: ./install_deps.sh
# ============================================

# ============================================
# CONFIGURATION
# ============================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="install_$(date +%Y%m%d_%H%M%S).log"

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

# ============================================
# CHECK VIRTUAL ENVIRONMENT
# ============================================

check_venv() {
    print_header "Checking Virtual Environment"
    
    if [ -z "$VIRTUAL_ENV" ]; then
        log_error "Not inside a virtual environment. Please activate your venv first"
    else
        log_success "Virtual environment detected: $VIRTUAL_ENV"
        PY_VERSION=$(python --version)
        log_info "Python: $PY_VERSION"
    fi
}

# ============================================
# SYSTEM DEPENDENCIES (apt only - no pip)
# ============================================

install_system_deps() {
    print_header "Installing System Dependencies (apt)"
    
    log "Updating package lists..."
    sudo apt update -y >> "$LOG_FILE" 2>&1
    log_success "Package lists updated"
    
    log "Installing required system packages..."
    
    PACKAGES=(
        "ffmpeg"
        "python3-opencv"
        "libatlas-base-dev"
        "libopenblas-dev"
    )
    
    for package in "${PACKAGES[@]}"; do
        log "  Installing $package..."
        sudo apt install -y $package >> "$LOG_FILE" 2>&1
        if [ $? -eq 0 ]; then
            log_success "    $package installed"
        else
            log_warning "    Failed to install $package"
        fi
    done
    
    log_success "System dependencies installation complete"
}

# ============================================
# PYTHON PACKAGES (pip only - no venv creation)
# ============================================

install_python_packages() {
    print_header "Installing Python Packages (pip)"
    
    # Ensure pip is available
    if ! command -v pip &> /dev/null; then
        log_error "pip not found. Please ensure pip is installed in your venv"
    fi
    
    log_info "Installing packages into existing venv: $VIRTUAL_ENV"
    
    # Install numpy first (required for everything)
    log "Installing numpy..."
    pip install numpy --no-cache-dir >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        log_success "numpy installed"
    else
        log_error "numpy installation failed"
    fi
    
    # Install psutil (system monitoring)
    log "Installing psutil..."
    pip install psutil --no-cache-dir >> "$LOG_FILE" 2>&1
    log_success "psutil installed"
    
    # Install ultralytics (includes YOLO and PyTorch dependencies)
    log "Installing ultralytics (this may take a while on RPi5)..."
    pip install ultralytics --no-cache-dir >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "ultralytics installed"
    else
        log_error "ultralytics installation failed"
    fi
    
    # Note: opencv is installed via apt (python3-opencv), not pip
    log_info "OpenCV is provided by system package (python3-opencv)"
    
    log_success "All Python packages installed"
}

# ============================================
# VERIFY INSTALLATIONS
# ============================================

verify_installations() {
    print_header "Verifying Installations"
    
    log "Testing imports..."
    
    # Test numpy
    python -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>> "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log_success "numpy import successful"
    else
        log_error "numpy import failed"
    fi
    
    # Test OpenCV (system package)
    python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" 2>> "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log_success "OpenCV import successful"
    else
        log_warning "OpenCV import failed - check python3-opencv installation"
    fi
    
    # Test psutil
    python -c "import psutil; print(f'  psutil: {psutil.__version__}')" 2>> "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log_success "psutil import successful"
    else
        log_warning "psutil import failed"
    fi
    
    # Test ultralytics (includes torch)
    python -c "from ultralytics import YOLO; print('  ultralytics: OK')" 2>> "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log_success "ultralytics import successful"
    else
        log_error "ultralytics import failed"
    fi
    
    # Show torch version (installed as ultralytics dependency)
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>> "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log_success "PyTorch import successful"
    fi
    
    log_success "All verifications passed"
}

# ============================================
# CHECK RASPBERRY PI HARDWARE (optional)
# ============================================

check_rpi_hardware() {
    print_header "Hardware Detection"
    
    # Check if running on Raspberry Pi
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model)
        log_success "Device: $MODEL"
        
        # Check CPU temperature (RPi specific)
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            TEMP=$(cat /sys/class/thermal/thermal_zone0/temp)
            TEMP_C=$((TEMP/1000))
            log_info "CPU Temperature: ${TEMP_C}°C"
        fi
    else
        log_info "Not running on Raspberry Pi (or model info not available)"
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    log_info "CPU Cores: $CPU_CORES"
    
    # Check RAM
    TOTAL_RAM=$(free -h | awk '/^Mem:/{print $2}')
    AVAIL_RAM=$(free -h | awk '/^Mem:/{print $7}')
    log_info "Total RAM: $TOTAL_RAM"
    log_info "Available RAM: $AVAIL_RAM"
    
    # Optional: Set CPU governor (uncomment if needed)
    # log "Setting CPU governor to performance mode..."
    # echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1
    # log_success "CPU performance mode enabled (requires sudo)"
}

# ============================================
# CHECK INPUT FILES (optional)
# ============================================

check_input_files() {
    print_header "Checking Input Files"
    
    if [ -f "best2.pt" ]; then
        MODEL_SIZE=$(du -h best2.pt | cut -f1)
        log_success "Model found: best2.pt ($MODEL_SIZE)"
    else
        log_warning "Model file 'best2.pt' not found (will need it for inference)"
    fi
    
    if [ -f "test2.mp4" ]; then
        VIDEO_SIZE=$(du -h test2.mp4 | cut -f1)
        log_success "Video found: test2.mp4 ($VIDEO_SIZE)"
    else
        log_warning "Video file 'test2.mp4' not found (will need it for inference)"
    fi
}

# ============================================
# PRINT USAGE INSTRUCTIONS
# ============================================

print_usage() {
    print_header "Next Steps"
    
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo -e "${CYAN}To run the inference script:${NC}"
    echo "  python run.py"
    echo ""
    echo -e "${CYAN}Or use the full pipeline with run directory:${NC}"
    echo "  python run.py  # Will create research_runs/run_XXX/output_with_detections.mp4"
    echo ""
    echo -e "${CYAN}Check that your YOLO model and video files are in the current directory:${NC}"
    echo "  ls -la best2.pt test2.mp4"
    echo ""
    echo -e "${YELLOW}Note:${NC} Make sure your Python script (run.py) is in the current directory"
    echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
}

# ============================================
# MAIN EXECUTION
# ============================================

main() {
    clear
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     YOLO INFERENCE - PACKAGE INSTALLER ONLY              ║"
    echo "║         Assumes existing virtual environment              ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "Starting package installation at $(date)"
    
    # Run installation steps (no venv creation, no pip upgrade)
    check_venv
    install_system_deps
    install_python_packages
    verify_installations
    check_rpi_hardware
    check_input_files
    print_usage
    
    log_success "Package installation completed at $(date)"
}

# ============================================
# RUN MAIN
# ============================================

main
