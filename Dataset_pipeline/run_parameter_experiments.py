"""
PARAMETER EXPERIMENT RUNNER
Runs experiments with different parameter combinations
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import torch

from optimized_translation_pipeline import main

# ---------------------------------------------------------
# LOAD COMET MODEL ONCE
# ---------------------------------------------------------

from comet import download_model, load_from_checkpoint

print("\nLoading COMET-KIWI model once for all experiments...")

MODEL_PATH = download_model("Unbabel/wmt22-cometkiwi-da")
COMET_MODEL = load_from_checkpoint(MODEL_PATH)

USE_GPU = 1 if torch.cuda.is_available() else 0

print(f"✓ COMET model ready (GPU: {USE_GPU})")


# ---------------------------------------------------------
# RUN SINGLE EXPERIMENT
# ---------------------------------------------------------

def run_single_experiment(experiment_id, params, base_checkpoint_dir='checkpoints'):

    checkpoint_dir = f"{base_checkpoint_dir}/exp_{experiment_id}"

    adaptive_params = {
        'base_weight': params['base_weight'],
        'z_weight': params['z_weight'],
        'z_threshold': params['z_threshold']
    }

    density_params = {
        'sparse_bonus': params['sparse_bonus'],
        'crowded_penalty': params['crowded_penalty'],
        'sparse_threshold': params['sparse_threshold'],
        'crowded_threshold': params['crowded_threshold']
    }

    score_weights = {
        'cosine_weight': params['cosine_weight'],
        'l2_weight': params['l2_weight']
    }

    try:

        result = main(
            malayalam_csv='malayalam_clean_samanantar.csv',
            hindi_csv='hindi_clean_samanantar.csv',
            search_k=int(params['search_k']),
            top_after_filtering=int(params['top_after_filtering']),
            comet_threshold=float(params['comet_threshold']),
            checkpoint_dir=checkpoint_dir,
            resume=True,
            auto_resume=True,
            adaptive_params=adaptive_params,
            density_params=density_params,
            score_weights=score_weights,

            # PASS GLOBAL COMET MODEL
            comet_model=COMET_MODEL,
            use_gpu=USE_GPU
        )

        if isinstance(result, dict):

            return {
                'experiment_id': experiment_id,
                'status': 'completed',

                'weighted_score': result.get('weighted_score', 0.0),
                'rank_1_percentage': result.get('rank_1_percentage', 0.0),
                'rank_2_percentage': result.get('rank_2_percentage', 0.0),
                'rank_3_percentage': result.get('rank_3_percentage', 0.0),

                'rank_1_count': result.get('rank_1_count', 0),
                'rank_2_count': result.get('rank_2_count', 0),
                'rank_3_count': result.get('rank_3_count', 0),

                'total_pairs': result.get('total_pairs', 0),
                'successful': result.get('successful', 0),
                'failed': result.get('failed', 0),

                'avg_selected_margin': result.get('avg_selected_margin', 0.0),
                'std_selected_margin': result.get('std_selected_margin', 0.0),
                'final_pairs': result.get('final_pairs', []),

                **params
            }

        return {'experiment_id': experiment_id, 'status': 'failed'}

    except Exception as e:

        import traceback

        print(f"\n❌ Error in experiment {experiment_id}: {e}")
        traceback.print_exc()

        return {
            'experiment_id': experiment_id,
            'status': 'failed',
            'error': str(e),
            **params
        }


# ---------------------------------------------------------
# RUN ALL EXPERIMENTS
# ---------------------------------------------------------

def run_experiments(parameter_file='parameter_experiments/parameters_minimal.csv',
                   results_file='experiment_results.csv',
                   start_from=1,
                   max_experiments=None):

    print("=" * 70)
    print("PARAMETER EXPERIMENT RUNNER")
    print("=" * 70)

    df_params = pd.read_csv(parameter_file)

    df_params = df_params[df_params['experiment_id'] >= start_from]

    if max_experiments:
        df_params = df_params.head(max_experiments)

    Path('results').mkdir(exist_ok=True)

    results = []
    best_score = -1
    best_experiment_id = None

    for _, row in tqdm(df_params.iterrows(), total=len(df_params), desc="Experiments"):

        experiment_id = int(row['experiment_id'])
        params = row.to_dict()

        print(f"\n🚀 Running Experiment {experiment_id}")

        result = run_single_experiment(experiment_id, params)
        final_pairs = result.pop('final_pairs', [])
        results.append(result)
        # ----------------------------------------
        # ⭐ CHECK FOR BEST EXPERIMENT
        # ----------------------------------------
        if result.get('status') == 'completed':

            current_score = result.get('weighted_score', 0.0)

            if current_score > best_score:

                print(f"\n🔥 NEW BEST EXPERIMENT: {experiment_id}")
                print(f"   Score: {current_score:.4f} (prev: {best_score:.4f})")

                best_score = current_score
                best_experiment_id = experiment_id

                # 👉 DIRECTLY SAVE final_pairs (NO RERUN)
                if final_pairs:

                    print("💾 Saving best experiment output (no rerun)...")

                    df_best = pd.DataFrame(final_pairs)
                    df_best.to_csv("/dev/shm/final_best_pairs.csv", index=False, encoding='utf-8')

                else:
                    print("⚠ Warning: final_pairs not found in result")
        
        # ----------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        pd.DataFrame(results).to_csv(results_file, index=False)

    print("\n✓ Results saved to:", results_file)
    print(f"🏆 Best Experiment: {best_experiment_id} with score {best_score:.4f}")
    df_final = pd.DataFrame(results)

    best_row = df_final.loc[df_final['weighted_score'].idxmax()]

    print("\n📊 BEST EXPERIMENT DETAILS:")
    print(best_row.to_string())

    return df_final


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--params', default='parameter_experiments/parameters_minimal.csv')
    parser.add_argument('--results', default='experiment_results.csv')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--max', type=int, default=None)

    args = parser.parse_args()

    run_experiments(
        parameter_file=args.params,
        results_file=args.results,
        start_from=args.start,
        max_experiments=args.max
    )
    
    
    
    
    
    
    
# """
# PARAMETER EXPERIMENT RUNNER
# Runs experiments with different parameter combinations and tracks weighted Pre-COMET Rank metric
# """

# import pandas as pd
# import json
# from pathlib import Path
# from tqdm import tqdm
# import sys
# import gc
# import torch
# from optimized_translation_pipeline import main, FAISSRetrieval

# def run_single_experiment(experiment_id, params, base_checkpoint_dir='checkpoints'):
#     """
#     Run a single experiment with given parameters

#     Args:
#         experiment_id: Experiment ID
#         params: Dictionary of parameters
#         base_checkpoint_dir: Base directory for checkpoints

#     Returns:
#         Dictionary with results including weighted_score
#     """
#     # Create experiment-specific checkpoint directory
#     checkpoint_dir = f"{base_checkpoint_dir}/exp_{experiment_id}"
#     output_file = f"results/experiment_{experiment_id}_pairs.csv"

#     # Prepare parameter dictionaries
#     adaptive_params = {
#         'base_weight': params['base_weight'],
#         'z_weight': params['z_weight'],
#         'z_threshold': params['z_threshold']
#     }

#     density_params = {
#         'sparse_bonus': params['sparse_bonus'],
#         'crowded_penalty': params['crowded_penalty'],
#         'sparse_threshold': params['sparse_threshold'],
#         'crowded_threshold': params['crowded_threshold']
#     }

#     score_weights = {
#         'cosine_weight': params['cosine_weight'],
#         'l2_weight': params['l2_weight']
#     }

#     try:
#         # Run the experiment
#         result = main(
#             malayalam_csv='malayalam_sentences_demo.csv',
#             hindi_csv='hindi_sentences_demo.csv',
#             search_k=int(params['search_k']),
#             top_after_filtering=int(params['top_after_filtering']),
#             comet_threshold=float(params['comet_threshold']),  # Experimental parameter
#             checkpoint_dir=checkpoint_dir,
#             resume=True,  # Enable resume for crash recovery
#             auto_resume=True,  # Auto-resume without user input
#             adaptive_params=adaptive_params,
#             density_params=density_params,
#             score_weights=score_weights
#         )

#         # Extract metrics
#         if isinstance(result, dict):
#             weighted_score = result.get('weighted_score', 0.0)
#             rank_1_percentage = result.get('rank_1_percentage', 0.0)
#             rank_2_percentage = result.get('rank_2_percentage', 0.0)
#             rank_3_percentage = result.get('rank_3_percentage', 0.0)
#             rank_1_count = result.get('rank_1_count', 0)
#             rank_2_count = result.get('rank_2_count', 0)
#             rank_3_count = result.get('rank_3_count', 0)
#             total_pairs = result.get('total_pairs', 0)
#             successful = result.get('successful', 0)
#             failed = result.get('failed', 0)
#             avg_selected_margin = result.get('avg_selected_margin', 0.0)
#             std_selected_margin = result.get('std_selected_margin', 0.0)
#         else:
#             # Fallback if result is not dict (old format)
#             weighted_score = 0.0
#             rank_1_percentage = 0.0
#             rank_2_percentage = 0.0
#             rank_3_percentage = 0.0
#             rank_1_count = 0
#             rank_2_count = 0
#             rank_3_count = 0
#             total_pairs = 0
#             successful = 0
#             failed = 0
#             avg_selected_margin = 0.0
#             std_selected_margin = 0.0

#         return {
#             'experiment_id': experiment_id,
#             'status': 'completed',
#             'weighted_score': weighted_score,  # MAIN METRIC
#             'rank_1_percentage': rank_1_percentage,
#             'rank_2_percentage': rank_2_percentage,
#             'rank_3_percentage': rank_3_percentage,
#             'rank_1_count': rank_1_count,
#             'rank_2_count': rank_2_count,
#             'rank_3_count': rank_3_count,
#             'total_pairs': total_pairs,
#             'successful': successful,
#             'failed': failed,
#             'avg_selected_margin': avg_selected_margin,
#             'std_selected_margin': std_selected_margin,
#             **params  # Include all parameters
#         }

#     except Exception as e:
#         print(f"\n❌ Error in experiment {experiment_id}: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             'experiment_id': experiment_id,
#             'status': 'failed',
#             'error': str(e),
#             'weighted_score': 0.0,
#             'avg_selected_margin': 0.0,
#             'std_selected_margin': 0.0,
#             **params
#         }

# def run_experiments(parameter_file='parameter_experiments/parameters_minimal.csv',
#                    results_file='experiment_results.csv',
#                    start_from=1,
#                    max_experiments=None):
#     """
#     Run all experiments from parameter file

#     Args:
#         parameter_file: Path to CSV with parameter combinations
#         results_file: Path to save results
#         start_from: Start from this experiment ID (for resuming)
#         max_experiments: Maximum number of experiments to run (None = all)
#     """
#     print("=" * 70)
#     print("PARAMETER EXPERIMENT RUNNER")
#     print("=" * 70)

#     # Load parameters
#     if not Path(parameter_file).exists():
#         print(f"\n❌ Parameter file not found: {parameter_file}")
#         print("Please run generate_parameter_datasets.py first")
#         return None

#     df_params = pd.read_csv(parameter_file)
#     print(f"\n✓ Loaded {len(df_params)} parameter combinations from {parameter_file}")

#     # Filter to start from
#     df_params = df_params[df_params['experiment_id'] >= start_from]

#     # Limit if specified
#     if max_experiments:
#         df_params = df_params.head(max_experiments)

#     print(f"→ Running {len(df_params)} experiments (starting from ID {start_from})")

#     # Create results directory
#     Path('results').mkdir(exist_ok=True)

#     # Load existing results if resuming
#     results = []
#     completed_ids = set()
#     if Path(results_file).exists():
#         df_existing = pd.read_csv(results_file)
#         results = df_existing.to_dict('records')
#         # Only track experiments that are actually completed (not failed)
#         completed_ids = {int(r['experiment_id']) for r in results if r.get('status') == 'completed'}
#         print(f"✓ Loaded {len(results)} existing results")
#         if len(completed_ids) > 0:
#             completed_list = sorted(completed_ids)
#             print(f"✓ Found {len(completed_ids)} completed experiments: {completed_list[:10]}{'...' if len(completed_ids) > 10 else ''}")
#         else:
#             print("  (No completed experiments found)")
#     else:
#         print("No existing results file found, starting fresh")

#     # Run experiments
#     print("\n" + "=" * 70)
#     print("RUNNING EXPERIMENTS")
#     print("=" * 70)
#     print("Metric: Weighted Score (Rank 1=1.0, Rank 2=0.7, Rank 3=0.5, Rank 4=0.3, Rank 5+=0.1)")
#     print("=" * 70)

#     for idx, row in tqdm(df_params.iterrows(), total=len(df_params), desc="Experiments"):
#         experiment_id = int(row['experiment_id'])

#         # Skip if already completed
#         if experiment_id in completed_ids:
#             print(f"\n⏭ Skipping experiment {experiment_id} (already completed)")
#             continue

#         print(f"\n{'='*70}")
#         print(f"EXPERIMENT {experiment_id}/{len(df_params)}")
#         print(f"{'='*70}")

#         params = row.to_dict()
#         result = run_single_experiment(experiment_id, params)
#         results.append(result)

#         # Clear GPU memory after each experiment
#         if torch.cuda.is_available():
#             print("Clearing GPU memory...")
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
#         gc.collect()
#         print("GPU memory cleared")

#         # Save results after each experiment
#         df_results = pd.DataFrame(results)
#         df_results.to_csv(results_file, index=False)

#         # Print current best
#         if len(results) > 0:
#             df_temp = pd.DataFrame(results)
#             completed = df_temp[df_temp['status'] == 'completed']
#             if len(completed) > 0:
#                 best = completed.loc[completed['weighted_score'].idxmax()]
#                 print(f"\n📊 Current best (weighted_score): Experiment {best['experiment_id']} "
#                       f"= {best['weighted_score']:.3f} "
#                       f"(Rank 1: {best.get('rank_1_percentage', 0):.1f}%, "
#                       f"Rank 2: {best.get('rank_2_percentage', 0):.1f}%, "
#                       f"Rank 3: {best.get('rank_3_percentage', 0):.1f}%)")
#                 if 'avg_selected_margin' in completed.columns:
#                     best_margin = completed.loc[completed['avg_selected_margin'].idxmax()]
#                     print(f"   Best (avg_selected_margin): Experiment {best_margin['experiment_id']} "
#                           f"= {best_margin['avg_selected_margin']:.4f} (std={best_margin.get('std_selected_margin', 0):.4f})")

#     # Final summary
#     print("\n" + "=" * 70)
#     print("EXPERIMENT SUMMARY")
#     print("=" * 70)

#     df_results = pd.DataFrame(results)
#     df_results.to_csv(results_file, index=False)

#     completed = df_results[df_results['status'] == 'completed']
#     if len(completed) > 0:
#         print(f"\n✓ Completed: {len(completed)} experiments")
#         print(f"❌ Failed: {len(df_results) - len(completed)} experiments")

#         # Top 10 results by weighted_score
#         top_10 = completed.nlargest(10, 'weighted_score')
#         print("\n" + "=" * 70)
#         print("TOP 10 EXPERIMENTS (by Weighted Score)")
#         print("=" * 70)
#         display_cols = ['experiment_id', 'weighted_score', 'avg_selected_margin', 'std_selected_margin',
#                        'rank_1_percentage', 'rank_2_percentage', 'rank_3_percentage', 'total_pairs',
#                        'base_weight', 'z_weight', 'sparse_bonus', 'crowded_penalty',
#                        'comet_threshold']
#         available_cols = [col for col in display_cols if col in top_10.columns]
#         print(top_10[available_cols].to_string(index=False))

#         # Best by avg_selected_margin (distinguishes correct sentences)
#         if 'avg_selected_margin' in completed.columns:
#             best_margin = completed.loc[completed['avg_selected_margin'].idxmax()]
#             print("\n" + "=" * 70)
#             print("BEST BY AVG SELECTED MARGIN (filter distinguishes correct best)")
#             print("=" * 70)
#             print(f"Experiment ID: {best_margin['experiment_id']}")
#             print(f"Avg selected margin: {best_margin['avg_selected_margin']:.4f}")
#             print(f"Std selected margin: {best_margin.get('std_selected_margin', 0):.4f}")

#         # Best experiment (by weighted score)
#         best = completed.loc[completed['weighted_score'].idxmax()]
#         print("\n" + "=" * 70)
#         print("🏆 BEST EXPERIMENT")
#         print("=" * 70)
#         print(f"Experiment ID: {best['experiment_id']}")
#         print(f"Weighted Score: {best['weighted_score']:.3f} (max: 1.0)")
#         print(f"Rank Distribution:")
#         print(f"  Rank 1: {best.get('rank_1_count', 0)} pairs ({best.get('rank_1_percentage', 0):.1f}%)")
#         print(f"  Rank 2: {best.get('rank_2_count', 0)} pairs ({best.get('rank_2_percentage', 0):.1f}%)")
#         print(f"  Rank 3: {best.get('rank_3_count', 0)} pairs ({best.get('rank_3_percentage', 0):.1f}%)")
#         print(f"Total Pairs: {best.get('total_pairs', 0)}")
#         print(f"\nParameters:")
#         print(f"  base_weight: {best['base_weight']}")
#         print(f"  z_weight: {best['z_weight']}")
#         print(f"  z_threshold: {best['z_threshold']}")
#         print(f"  cosine_weight: {best['cosine_weight']}")
#         print(f"  l2_weight: {best['l2_weight']}")
#         print(f"  sparse_bonus: {best['sparse_bonus']}")
#         print(f"  crowded_penalty: {best['crowded_penalty']}")
#         print(f"  sparse_threshold: {best['sparse_threshold']}")
#         print(f"  crowded_threshold: {best['crowded_threshold']}")
#         print(f"  search_k: {best['search_k']}")
#         print(f"  top_after_filtering: {best['top_after_filtering']}")
#         print(f"  comet_threshold: {best['comet_threshold']}")

#     print(f"\n✓ Results saved to: {results_file}")
#     return df_results

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Run parameter experiments')
#     parser.add_argument('--params', type=str, default='parameter_experiments/parameters_minimal.csv')
#     parser.add_argument('--results', type=str, default='experiment_results.csv')
#     parser.add_argument('--start', type=int, default=1)
#     parser.add_argument('--max', type=int, default=None)
    
#     args = parser.parse_args()
    
#     run_experiments(
#         parameter_file=args.params,
#         results_file=args.results,
#         start_from=args.start,
#         max_experiments=args.max
#     )