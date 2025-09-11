#!/usr/bin/env python3
"""
Command-line interface for inverter-predictive-maintenance package.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .preprocess import load_parquet_data, load_failure_sessions, prepare_dataset
from .visualize import visualize_failure_timeline, visualize_mean_values


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inverter Predictive Maintenance CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and visualize data
  inverter-predictive-maintenance visualize --data-path dataset/inverter_data --failures dataset/failures.csv
  
  # Prepare dataset for training
  inverter-predictive-maintenance prepare --data-path dataset/inverter_data --failures dataset/failures.csv --output dataset/processed/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize inverter data')
    viz_parser.add_argument('--data-path', required=True, help='Path to inverter data directory')
    viz_parser.add_argument('--failures', required=True, help='Path to failure sessions CSV')
    viz_parser.add_argument('--output', default='plot/', help='Output directory for plots')
    viz_parser.add_argument('--features', nargs='+', help='Feature columns to visualize')
    
    # Prepare command
    prep_parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
    prep_parser.add_argument('--data-path', required=True, help='Path to inverter data directory')
    prep_parser.add_argument('--failures', required=True, help='Path to failure sessions CSV')
    prep_parser.add_argument('--output', required=True, help='Output directory for processed data')
    prep_parser.add_argument('--pre-days', type=int, default=5, help='Pre-failure labeling days')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'visualize':
            run_visualization(args)
        elif args.command == 'prepare':
            run_preparation(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_visualization(args):
    """Run data visualization."""
    print("Loading data for visualization...")
    
    # Load data
    inverter_data = load_parquet_data(args.data_path)
    failure_sessions = load_failure_sessions(args.failures)
    
    print(f"Loaded {len(inverter_data)} inverter records and {len(failure_sessions)} failure sessions")
    
    # Default features if not specified
    if not args.features:
        args.features = [
            "metric.STATUS_AC_MOD_ADMISSION_TEMP.MEASURED",
            "metric.STATUS_INTERNAL_TEMP.MEASURED",
            "metric.AC_VOLTAGE_AB.MEASURED",
            "metric.AC_VOLTAGE_BC.MEASURED",
            "metric.AC_VOLTAGE_CA.MEASURED",
            "metric.DC_VOLTAGE.MEASURED",
            "metric.AC_POWER.MEASURED",
        ]
    
    # Create visualizations
    print("Creating failure timeline...")
    visualize_failure_timeline(failure_sessions)
    
    print("Creating time series visualizations...")
    visualize_mean_values(
        inverter_data=inverter_data,
        failure_sessions=failure_sessions,
        feature_cols=args.features,
        output_dir=args.output,
        title="Inverter Data Visualization"
    )
    
    print(f"Visualizations saved to {args.output}")


def run_preparation(args):
    """Run dataset preparation."""
    print("Preparing dataset...")
    
    # Load data
    inverter_data = load_parquet_data(args.data_path)
    failure_sessions = load_failure_sessions(args.failures)
    
    print(f"Loaded {len(inverter_data)} inverter records and {len(failure_sessions)} failure sessions")
    
    # Prepare dataset
    prepared_data = prepare_dataset(
        inverter_data=inverter_data,
        failure_sessions=failure_sessions,
        pre_days=args.pre_days
    )
    
    # Save prepared data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prepared_data.to_csv(output_path / "prepared_data.csv", index=False)
    
    print(f"Prepared dataset saved to {output_path / 'prepared_data.csv'}")
    print(f"Dataset shape: {prepared_data.shape}")


if __name__ == "__main__":
    main()
