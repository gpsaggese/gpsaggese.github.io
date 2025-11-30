import matplotlib.pyplot as plt
from pathlib import Path

def save_plot(filename, subdirectory="results"):
    """Save plot to specified subdirectory"""
    import os
    from pathlib import Path
    
    # Create directory if it doesn't exist
    output_dir = Path(subdirectory)
    output_dir.mkdir(exist_ok=True)
    
    # Save plot
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filepath}")

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")