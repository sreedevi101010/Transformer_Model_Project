"""
PARAMETER EXPERIMENT GENERATOR (No dependencies - uses only standard library)
Generates all combinations of test parameters for experimentation
"""

import itertools
import csv
from pathlib import Path

def generate_minimal_combinations():
    """
    Generate minimal parameter combinations (smallest set for quick testing)
    Focuses on parameters with major impact
    """
    parameters = {
        # Adaptive Scoring - Most Impact (z_weight will be calculated as 1.0 - base_weight)
        'base_weight': [0.6, 0.7, 0.8],

        # Density - Most Impact
        'sparse_bonus': [0.01, 0.10, 0.3],
        'crowded_penalty': [0.01, 0.15, 0.3],

        # Fixed values
        'z_threshold': [-1.0],
        'cosine_weight': [0.6, 0.7, 0.8],
        'sparse_threshold': [0.1, 0.3],
        'crowded_threshold': [0.1, 0.3],
        'search_k': [100],
        'top_after_filtering': [5],  # Top 5 candidates go to COMET
        'comet_threshold': [0.70],  # COMET quality threshold
    }

    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    all_combinations = list(itertools.product(*param_values))

    # Convert to list of dictionaries
    rows = []
    for i, combo in enumerate(all_combinations, start=1):
        row = dict(zip(param_names, combo))
        # Calculate z_weight as complement of base_weight
        row['z_weight'] = round(1.0 - row['base_weight'], 2)
        row['l2_weight'] = round(1.0 - row['cosine_weight'], 2)
        row['experiment_id'] = i
        rows.append(row)

    # Define column order
    column_order = [
        'experiment_id',
        'search_k',
        'top_after_filtering',
        'comet_threshold',
        'cosine_weight',
        'l2_weight',
        'base_weight',
        'z_weight',
        'z_threshold',
        'sparse_bonus',
        'crowded_penalty',
        'sparse_threshold',
        'crowded_threshold',
    ]

    return rows, column_order

def generate_focused_combinations():
    """
    Generate focused parameter combinations (medium set for comprehensive testing)
    """
    parameters = {
        # Adaptive Scoring (z_weight will be calculated as 1.0 - base_weight)
        'base_weight': [0.5, 0.6, 0.7, 0.8],
        'z_threshold': [-1.5, -1.0, -0.5],

        # Combined Score Weights
        'cosine_weight': [0.5, 0.6, 0.7,0.8],

        # Density Parameters
        'sparse_bonus': [0.05, 0.10, 0.15],
        'crowded_penalty': [0.10, 0.15, 0.20],

        # Fixed values
        'sparse_threshold': [0.3],
        'crowded_threshold': [0.1],
        'search_k': [100],
        'top_after_filtering': [5],  # Top 5 candidates go to COMET
        'comet_threshold': [0.50, 0.75],  # COMET quality threshold
    }

    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    all_combinations = list(itertools.product(*param_values))

    # Convert to list of dictionaries
    rows = []
    for i, combo in enumerate(all_combinations, start=1):
        row = dict(zip(param_names, combo))
        # Calculate z_weight as complement of base_weight
        row['z_weight'] = round(1.0 - row['base_weight'], 2)
        row['l2_weight'] = round(1.0 - row['cosine_weight'], 2)
        row['experiment_id'] = i
        rows.append(row)

    # Define column order
    column_order = [
        'experiment_id',
        'search_k',
        'top_after_filtering',
        'comet_threshold',
        'cosine_weight',
        'l2_weight',
        'base_weight',
        'z_weight',
        'z_threshold',
        'sparse_bonus',
        'crowded_penalty',
        'sparse_threshold',
        'crowded_threshold',
    ]

    return rows, column_order

def generate_baseline_experiments():
    """
    Generate 3 baseline experiments:
    1. No adaptive (adaptive disabled, density enabled)
    2. No density (adaptive enabled, density disabled)
    3. Both disabled (baseline - just cosine+L2, no adaptive, no density)
    """
    # Base parameters (same for all baseline experiments)
    base_params = {
        'search_k': 100,
        'top_after_filtering': 5,
        'comet_threshold': 0.70,
        'cosine_weight': 0.6,
        'l2_weight': 0.4,
        'z_threshold': -1.0,
        'sparse_threshold': 0.3,
        'crowded_threshold': 0.1,
    }

    rows = []

    # Use high starting ID (10000+) to avoid conflicts with regular experiments
    baseline_start_id = 10000

    # Experiment 1: No Adaptive (density enabled)
    row1 = base_params.copy()
    row1.update({
        'experiment_id': baseline_start_id + 1,
        'base_weight': 1.0,  # Disable adaptive: base_weight=1.0 means adaptive_score = score
        'z_weight': 0.0,
        'sparse_bonus': 0.1,  # Density enabled
        'crowded_penalty': 0.15,
    })
    rows.append(row1)

    # Experiment 2: No Density (adaptive enabled)
    row2 = base_params.copy()
    row2.update({
        'experiment_id': baseline_start_id + 2,
        'base_weight': 0.6,  # Adaptive enabled
        'z_weight': 0.4,
        'sparse_bonus': 0.0,  # Disable density: no bonus/penalty
        'crowded_penalty': 0.0,
    })
    rows.append(row2)

    # Experiment 3: Both Disabled (baseline)
    row3 = base_params.copy()
    row3.update({
        'experiment_id': baseline_start_id + 3,
        'base_weight': 1.0,  # Disable adaptive
        'z_weight': 0.0,
        'sparse_bonus': 0.0,  # Disable density
        'crowded_penalty': 0.0,
    })
    rows.append(row3)

    # Column order (same as other functions)
    column_order = [
        'experiment_id',
        'search_k',
        'top_after_filtering',
        'comet_threshold',
        'cosine_weight',
        'l2_weight',
        'base_weight',
        'z_weight',
        'z_threshold',
        'sparse_bonus',
        'crowded_penalty',
        'sparse_threshold',
        'crowded_threshold',
    ]

    return rows, column_order

if __name__ == "__main__":
    output_dir = Path('parameter_experiments')
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GENERATING PARAMETER EXPERIMENT DATASETS")
    print("=" * 70)

    # Generate minimal combinations
    print("\n1. Generating MINIMAL combinations...")
    rows_minimal, column_order = generate_minimal_combinations()
    print(f"   ✓ Generated {len(rows_minimal)} combinations")

    minimal_file = output_dir / 'parameters_minimal.csv'
    with open(minimal_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(rows_minimal)
    print(f"   ✓ Saved to {minimal_file}")

    # Generate focused combinations
    print("\n2. Generating FOCUSED combinations...")
    rows_focused, column_order = generate_focused_combinations()
    print(f"   ✓ Generated {len(rows_focused)} combinations")

    focused_file = output_dir / 'parameters_focused.csv'
    with open(focused_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
        writer.writerows(rows_focused)
    print(f"   ✓ Saved to {focused_file}")

    # Generate baseline experiments (no adaptive, no density, both disabled)
    print("\n3. Generating BASELINE experiments...")
    rows_baseline, baseline_column_order = generate_baseline_experiments()
    print(f"   ✓ Generated {len(rows_baseline)} baseline experiments")

    # Save baseline experiments to separate CSV file
    baseline_file = output_dir / 'parameters_baseline.csv'
    with open(baseline_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=baseline_column_order)
        writer.writeheader()
        writer.writerows(rows_baseline)
    print(f"   ✓ Saved to {baseline_file}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Minimal dataset:  {len(rows_minimal):,} combinations")
    print(f"Focused dataset:  {len(rows_focused):,} combinations")
    print(f"Baseline dataset: {len(rows_baseline)} experiments")
    print(f"\nBaseline experiments:")
    print(f"  1. No Adaptive (density enabled) - ID: 10001")
    print(f"  2. No Density (adaptive enabled) - ID: 10002")
    print(f"  3. Both Disabled (baseline) - ID: 10003")
    print(f"\nAll files saved to: {output_dir.absolute()}")

    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE FROM MINIMAL DATASET (first 5 rows)")
    print("=" * 70)
    for i, row in enumerate(rows_minimal[:5], 1):
        print(f"\nRow {i}:")
        for key in column_order:
            print(f"  {key}: {row[key]}")

    print("\n" + "=" * 70)
    print("✓ PARAMETER DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the generated CSV files in 'parameter_experiments/' directory")
    print("2. Start with 'parameters_minimal.csv' for quick testing")
    print("3. Run baseline experiments: python3 run_parameter_experiments.py --params parameter_experiments/parameters_baseline.csv")
    print("4. Run full experiments: python3 run_parameter_experiments.py --params parameter_experiments/parameters_minimal.csv")
