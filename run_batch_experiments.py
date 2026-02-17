#!/usr/bin/env python3
"""
Batch Experiment Runner for GTCC, TCC, VAVA, and LAV methods.

This script runs all 4 methods sequentially using 8 GPUs each with DDP.
Output is stored in output_batch/ folder with auto-incrementing version numbers.

Usage:
    python run_batch_experiments.py                    # Run with default batch size
    python run_batch_experiments.py -bs 8              # Run with batch size 8
    python run_batch_experiments.py --dry-run          # Show commands without running
    python run_batch_experiments.py --methods gtcc tcc # Run only specific methods
"""

import os
import re
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


# Configuration
METHODS = ['gtcc', 'tcc', 'vava', 'lav']
NUM_GPUS = 8
DEFAULT_OUTPUT_DIR = 'output_batch'
TRAIN_SCRIPT = 'multitask_train.py'


def get_next_version(output_dir: str) -> int:
    """
    Scan the output directory for existing V{N}_ prefixes and return N+1.

    Args:
        output_dir: Path to the output directory

    Returns:
        Next version number (starts at 1 if no existing versions)
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return 1

    # Pattern to match V{N}_ prefix in folder names
    version_pattern = re.compile(r'^V(\d+)_')

    max_version = 0
    for item in output_path.iterdir():
        if item.is_dir():
            match = version_pattern.match(item.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)

    return max_version + 1


def build_command(method: str, version: int, batch_size: int = None,
                  epochs: int = None, output_dir: str = DEFAULT_OUTPUT_DIR) -> list:
    """
    Build the torchrun command for a given method.

    Args:
        method: One of 'gtcc', 'tcc', 'vava', 'lav'
        version: Version number for the experiment
        batch_size: Optional batch size override
        epochs: Optional epochs override
        output_dir: Output directory path

    Returns:
        List of command arguments
    """
    cmd = [
        'torchrun',
        f'--nproc_per_node={NUM_GPUS}',
        TRAIN_SCRIPT,
        str(version),  # Just the number, code adds 'V' prefix
        f'--{method}',
        '--ego',
        '--resnet',
        '--mcn',
    ]

    if batch_size is not None:
        cmd.extend(['-bs', str(batch_size)])

    if epochs is not None:
        cmd.extend(['-ep', str(epochs)])

    return cmd


def run_method(method: str, version: int, batch_size: int = None,
               epochs: int = None, output_dir: str = DEFAULT_OUTPUT_DIR,
               dry_run: bool = False) -> tuple:
    """
    Run a single method and return success status.

    Args:
        method: Method name
        version: Version number
        batch_size: Optional batch size
        epochs: Optional epochs
        output_dir: Output directory
        dry_run: If True, just print the command

    Returns:
        Tuple of (success: bool, duration_seconds: float)
    """
    cmd = build_command(method, version, batch_size, epochs, output_dir)
    cmd_str = ' '.join(cmd)

    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"Version: V{version}")
    print(f"Command: {cmd_str}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return True, 0.0

    # Set OUTPUT_PATH environment variable to use output_batch directory
    env = os.environ.copy()
    env['OUTPUT_PATH'] = output_dir

    start_time = time.time()

    try:
        # Run the command and stream output
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        process.wait()
        duration = time.time() - start_time

        if process.returncode == 0:
            print(f"\n[SUCCESS] {method.upper()} completed in {duration/60:.1f} minutes")
            return True, duration
        else:
            print(f"\n[FAILED] {method.upper()} exited with code {process.returncode}")
            return False, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] {method.upper()} failed with exception: {e}")
        return False, duration


def main():
    parser = argparse.ArgumentParser(
        description='Run batch experiments for GTCC, TCC, VAVA, and LAV methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_batch_experiments.py                    # Run all methods with default settings
    python run_batch_experiments.py -bs 8              # Run with batch size 8
    python run_batch_experiments.py --dry-run          # Show commands without running
    python run_batch_experiments.py --methods gtcc lav # Run only GTCC and LAV
        """
    )

    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=None,
        help='Batch size per GPU (overrides default in parser_util.py)'
    )

    parser.add_argument(
        '-ep', '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides default)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        choices=METHODS,
        default=METHODS,
        help='Methods to run (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )

    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='Force a specific version number (default: auto-detect)'
    )

    args = parser.parse_args()

    # Determine version
    if args.version is not None:
        version = args.version
    else:
        version = get_next_version(args.output_dir)

    # Create output directory if it doesn't exist
    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n" + "="*60)
    print("BATCH EXPERIMENT RUNNER")
    print("="*60)
    print(f"Version: V{version}")
    print(f"Methods: {', '.join(m.upper() for m in args.methods)}")
    print(f"GPUs per method: {NUM_GPUS}")
    print(f"Batch size: {args.batch_size if args.batch_size else 'default'}")
    print(f"Epochs: {args.epochs if args.epochs else 'default'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dry run: {args.dry_run}")
    print("="*60 + "\n")

    # Run each method sequentially
    results = {}
    total_start = time.time()

    for method in args.methods:
        success, duration = run_method(
            method=method,
            version=version,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )
        results[method] = {'success': success, 'duration': duration}

    total_duration = time.time() - total_start

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Version: V{version}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print("\nResults:")

    for method, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        duration_str = f"{result['duration']/60:.1f} min" if result['duration'] > 0 else "N/A"
        print(f"  {method.upper():6s}: {status:8s} ({duration_str})")

    # Check if all succeeded
    all_success = all(r['success'] for r in results.values())

    print("\n" + "="*60)
    if all_success:
        print("All experiments completed successfully!")
    else:
        failed = [m for m, r in results.items() if not r['success']]
        print(f"Some experiments failed: {', '.join(f.upper() for f in failed)}")
    print("="*60 + "\n")

    # Return exit code
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
