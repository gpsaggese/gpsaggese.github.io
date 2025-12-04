#!/bin/bash

# Fake News Detection - Docker Entrypoint Script
# This script orchestrates the entire project execution

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Directory setup
log_info "Setting up directories..."
mkdir -p /app/data /app/models /app/output /app/logs

# NLTK data setup
log_info "Setting up NLTK data..."
python3 << 'PYTHON_EOF'
import nltk
import os
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')
PYTHON_EOF

# Function definitions
run_accuracy_tests() {
    log_info "Running accuracy tests..."
    log_info "=================================================="
    python3 test_accuracy_simple.py 2>&1 | tee /app/logs/accuracy_tests_$(date +%Y%m%d_%H%M%S).log
    log_success "Accuracy tests completed!"
}

run_bert_training() {
    log_info "Running BERT training pipeline..."
    log_info "=================================================="
    if [ -f "train_bert_liar_only.py" ]; then
        python3 train_bert_liar_only.py 2>&1 | tee /app/logs/bert_training_$(date +%Y%m%d_%H%M%S).log
        log_success "BERT training completed!"
    else
        log_warning "BERT training script not found"
    fi
}

run_lstm_training() {
    log_info "Running LSTM training pipeline..."
    log_info "=================================================="
    if [ -f "train_optimized.py" ]; then
        python3 train_optimized.py 2>&1 | tee /app/logs/lstm_training_$(date +%Y%m%d_%H%M%S).log
        log_success "LSTM training completed!"
    else
        log_warning "LSTM training script not found"
    fi
}

run_cross_validation() {
    log_info "Running k-fold cross-validation..."
    log_info "=================================================="
    if [ -f "cross_validation.py" ]; then
        python3 cross_validation.py 2>&1 | tee /app/logs/cross_validation_$(date +%Y%m%d_%H%M%S).log
        log_success "Cross-validation completed!"
    else
        log_warning "Cross-validation script not found"
    fi
}

run_mcp_server() {
    log_info "Starting MCP Server..."
    log_info "=================================================="
    python3 MCP.server.py 2>&1 | tee /app/logs/mcp_server_$(date +%Y%m%d_%H%M%S).log &
    MCP_PID=$!
    log_success "MCP Server started (PID: $MCP_PID)"
}

run_enhanced_training() {
    log_info "Running enhanced training pipeline..."
    log_info "=================================================="
    if [ -f "enhanced_training.py" ]; then
        python3 enhanced_training.py 2>&1 | tee /app/logs/enhanced_training_$(date +%Y%m%d_%H%M%S).log
        log_success "Enhanced training completed!"
    else
        log_warning "Enhanced training script not found"
    fi
}

run_all_evaluations() {
    log_info "Running all model evaluations..."
    log_info "=================================================="
    if [ -f "evaluate_all_models.py" ]; then
        python3 evaluate_all_models.py 2>&1 | tee /app/logs/all_evaluations_$(date +%Y%m%d_%H%M%S).log
        log_success "All evaluations completed!"
    else
        log_warning "Evaluation script not found"
    fi
}

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║         FAKE NEWS DETECTION - DOCKER EXECUTION                    ║"
    echo "║                                                                    ║"
    echo "║  Project: TutorTask144_Fall2025_Fake_News_Detection               ║"
    echo "║  Date: $(date)                               ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_menu() {
    echo ""
    echo "Select execution mode:"
    echo ""
    echo "  1) Run all tests and training (FULL EXECUTION)"
    echo "  2) Run accuracy tests only"
    echo "  3) Run BERT training only"
    echo "  4) Run LSTM training only"
    echo "  5) Run cross-validation only"
    echo "  6) Run enhanced training only"
    echo "  7) Run all evaluations only"
    echo "  8) Start MCP server (interactive)"
    echo "  9) Start interactive bash shell"
    echo "  0) Exit"
    echo ""
}

# Main execution logic
main() {
    local mode="${1:-all}"
    
    print_header
    
    case "$mode" in
        "all")
            log_info "Starting FULL PROJECT EXECUTION..."
            log_info "Executing all pipelines..."
            echo ""
            
            run_accuracy_tests
            echo ""
            
            run_cross_validation
            echo ""
            
            run_bert_training
            echo ""
            
            run_lstm_training
            echo ""
            
            run_enhanced_training
            echo ""
            
            run_all_evaluations
            echo ""
            
            log_success "════════════════════════════════════════════════════════════"
            log_success "ALL PIPELINES COMPLETED SUCCESSFULLY!"
            log_success "════════════════════════════════════════════════════════════"
            log_info "Logs saved to: /app/logs/"
            log_info "Results available in: /app/output/"
            ;;
            
        "test")
            run_accuracy_tests
            ;;
            
        "bert")
            run_bert_training
            ;;
            
        "lstm")
            run_lstm_training
            ;;
            
        "cv")
            run_cross_validation
            ;;
            
        "enhanced")
            run_enhanced_training
            ;;
            
        "eval")
            run_all_evaluations
            ;;
            
        "mcp")
            run_mcp_server
            log_info "MCP server is running. Press Ctrl+C to stop."
            wait
            ;;
            
        "bash"|"shell")
            log_info "Starting interactive bash shell..."
            /bin/bash
            ;;
            
        *)
            print_menu
            read -p "Enter selection (0-9): " choice
            case "$choice" in
                "1") main "all" ;;
                "2") main "test" ;;
                "3") main "bert" ;;
                "4") main "lstm" ;;
                "5") main "cv" ;;
                "6") main "enhanced" ;;
                "7") main "eval" ;;
                "8") main "mcp" ;;
                "9") main "bash" ;;
                "0") log_info "Exiting..."; exit 0 ;;
                *) log_error "Invalid selection"; main ;;
            esac
            ;;
    esac
}

# Run main
main "$@"
