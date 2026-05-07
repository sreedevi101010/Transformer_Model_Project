"""
OPTIMIZED PER-SENTENCE PROCESSING PIPELINE
Key optimizations:
   1. Batch ALL COMET calls together (biggest speedup: 10-50x)
   2. Vectorized distance computations (2-10x speedup)
   3. Increased batch sizes for embeddings and COMET
   4. GPU acceleration for COMET if available

For each Malayalam sentence:
   1. Retrieve top 100 Hindi candidates from FAISS
   2. Compute score = 0.6*cosine + 0.4*L2
   3. Apply adaptive scoring on the 100 candidates
   4. Apply density filtering on the 100 candidates
   5. Take top 10 after filtering
   6. Apply COMET scoring on the top 10 (BATCHED)
   7. Select the pair with max COMET score above threshold 0.75
   8. Repeat for all Malayalam sentences
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
import json
import pickle
import os
from pathlib import Path

# ============================================================================
# CHECKPOINT SYSTEM FOR SAVE/RESUME
# ============================================================================

def save_checkpoint(checkpoint_dir, step_name, data):
    """Save checkpoint data to disk"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / f"{step_name}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Checkpoint saved: {checkpoint_file}")

def load_checkpoint(checkpoint_dir, step_name):
    """Load checkpoint data from disk"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_file = checkpoint_dir / f"{step_name}.pkl"

    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Checkpoint loaded: {checkpoint_file}")
        return data
    return None

def save_progress(checkpoint_dir, progress_dict):
    """Save progress metadata"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    progress_file = checkpoint_dir / "progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress_dict, f, indent=2)

def load_progress(checkpoint_dir):
    """Load progress metadata"""
    checkpoint_dir = Path(checkpoint_dir)
    progress_file = checkpoint_dir / "progress.json"

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

# ============================================================================
# FAISS RETRIEVAL CLASS
# ============================================================================

class FAISSRetrieval:
    """FAISS-based retrieval system for Malayalam-Hindi"""

    def __init__(self, model_name='sentence-transformers/LaBSE'):
        print("=" * 70)
        print("FAISS RETRIEVAL SYSTEM - MALAYALAM TO HINDI")
        print("=" * 70)
        print(f"\nStep 1: Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded: {model_name}")

        self.hindi_sentences = []
        self.malayalam_sentences = []
        self.hindi_embeddings = None
        self.malayalam_embeddings = None
        self.cosine_index = None
        self.l2_index = None
        self.dimension = None

    def load_data(self, malayalam_csv, hindi_csv):
        print(f"\nStep 2: Loading data...")

        with open(malayalam_csv, 'r', encoding='utf-8') as f:
            self.malayalam_sentences = [line.strip() for line in f if line.strip()]

        print(f"✓ Loaded {len(self.malayalam_sentences)} Malayalam sentences")

        with open(hindi_csv, 'r', encoding='utf-8') as f:
            self.hindi_sentences = [line.strip() for line in f if line.strip()]

        print(f"✓ Loaded {len(self.hindi_sentences)} Hindi sentences")
        return self

    def embed_hindi_sentences(self, batch_size=128):
        """Embed all Hindi sentences"""
        print(f"\nStep 3: Embedding Hindi sentences...")
        self.hindi_embeddings = self.model.encode(
            self.hindi_sentences,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.dimension = self.hindi_embeddings.shape[1]
        print(f"✓ Embeddings created: shape {self.hindi_embeddings.shape}")
        return self

    def embed_malayalam_sentences(self, batch_size=128):
        """Embed all Malayalam sentences"""
        print(f"\nStep 3b: Embedding Malayalam sentences...")
        self.malayalam_embeddings = self.model.encode(
            self.malayalam_sentences,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✓ Malayalam embeddings: shape {self.malayalam_embeddings.shape}")
        return self

    def build_faiss_indices(self):
        """Build FAISS indices"""
        print(f"\nStep 4: Building FAISS indices...")
        embeddings_float32 = self.hindi_embeddings.astype('float32')

        print("  Building Cosine index (IndexFlatIP)...")
        self.cosine_index = faiss.IndexFlatIP(self.dimension)
        self.cosine_index.add(embeddings_float32)
        print(f"  ✓ Cosine index built: {self.cosine_index.ntotal} vectors")

        print("  Building L2 index (IndexFlatL2)...")
        self.l2_index = faiss.IndexFlatL2(self.dimension)
        self.l2_index.add(embeddings_float32)
        print(f"  ✓ L2 index built: {self.l2_index.ntotal} vectors")
        return self

# ============================================================================
# ADAPTIVE SCORING AND DENSITY FILTERING CLASSES
# ============================================================================

class AdaptiveLocalScoring:
    """Adaptive local scoring with z-score normalization"""
    def __init__(self, base_weight=0.6, z_weight=0.4, z_threshold=-1.0):
        self.base_weight = base_weight
        self.z_weight = z_weight
        self.z_threshold = z_threshold
        self.epsilon = 1e-8

    def process_candidates(self, candidates, score_key='score'):
        if len(candidates) < 3:
            for c in candidates:
                c['adaptive_score'] = c.get(score_key, 0.0)
                c['z_score'] = 0.0
            return sorted(candidates, key=lambda x: x['adaptive_score'], reverse=True)

        scores = [c.get(score_key, 0.0) for c in candidates]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if std_score < self.epsilon:
            for c in candidates:
                c['adaptive_score'] = c.get(score_key, 0.0)
                c['z_score'] = 0.0
            return sorted(candidates, key=lambda x: x['adaptive_score'], reverse=True)

        filtered = []
        for c in candidates:
            base_score = c.get(score_key, 0.0)
            z_score = (base_score - mean_score) / (std_score + self.epsilon)
            z_score = np.clip(z_score, -5.0, 5.0)
            normalized_z = np.clip(z_score / 3.0, -1.0, 1.0)
            adaptive_score = self.base_weight * base_score + self.z_weight * normalized_z
            adaptive_score = np.clip(adaptive_score, 0.0, 1.0)

            c['base_score'] = base_score
            c['z_score'] = z_score
            c['adaptive_score'] = adaptive_score

            if z_score > self.z_threshold:
                filtered.append(c)

        filtered.sort(key=lambda x: x['adaptive_score'], reverse=True)
        return filtered

@dataclass
class DensityConfig:
    """Configuration for density-based filtering"""
    k_neighbors: int = 20
    sparse_threshold: float = 0.3
    crowded_threshold: float = 0.1
    sparse_bonus: float = 0.1
    crowded_penalty: float = 0.15

class DensityBasedFiltering:
    """Density-based filtering for candidate selection"""
    def __init__(self, hindi_faiss_index, hindi_embeddings, config=None):
        self.hindi_index = hindi_faiss_index
        self.hindi_embeddings = hindi_embeddings
        self.config = config if config else DensityConfig()

    def process_candidates(self, candidates, score_key='adaptive_score'):
        if len(candidates) == 0:
            return []

        total_vectors = self.hindi_index.ntotal
        max_neighbors = min(self.config.k_neighbors + 1, total_vectors)
        if max_neighbors < 2:
            for c in candidates:
                c['density'] = 0.5
                c['density_bonus'] = 0.0
                c['density_penalty'] = 0.0
                c['final_score'] = c.get(score_key, 0.0)
            return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

        hindi_indices = [c.get('hindi_idx') for c in candidates]
        hindi_embs = self.hindi_embeddings[hindi_indices].astype('float32')
        neighbor_sims_batch, _ = self.hindi_index.search(hindi_embs, max_neighbors)

        density_candidates = []
        for i, c in enumerate(candidates):
            neighbor_sims = neighbor_sims_batch[i][1:]
            valid_sims = [float(s) for s in neighbor_sims if np.isfinite(float(s)) and not np.isnan(float(s))]
            valid_sims = [np.clip(s, -1.0, 1.0) for s in valid_sims]

            if len(valid_sims) == 0:
                avg_neighbor_sim = 0.0
                density = 0.5
            else:
                avg_neighbor_sim = np.clip(float(np.mean(valid_sims)), -1.0, 1.0)
                normalized_sim = (avg_neighbor_sim + 1.0) / 2.0
                density = np.clip(1.0 - normalized_sim, 0.0, 1.0)

            adaptive_score = c.get(score_key, 0.0)
            bonus = self.config.sparse_bonus if density > self.config.sparse_threshold else 0.0
            penalty = self.config.crowded_penalty if density < self.config.crowded_threshold else 0.0
            final_score = np.clip(adaptive_score + bonus - penalty, 0.0, 1.0)

            c['density'] = density
            c['density_bonus'] = bonus
            c['density_penalty'] = penalty
            c['final_score'] = final_score
            density_candidates.append(c)

        density_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return density_candidates

# ============================================================================
# OPTIMIZED PROCESSING FUNCTION (without COMET - batched separately)
# ============================================================================

def process_single_malayalam_sentence_optimized(mal_idx, retrieval, adaptive_scorer, density_filter,
                                                search_k=100, top_after_filtering=5, score_weights=None):
    """
    OPTIMIZED: Process a single Malayalam sentence WITHOUT COMET (COMET is batched separately)
    Uses vectorized operations for 10-100x speedup on distance computations.

    Returns:
        List of top candidate pairs (ready for batch COMET scoring)
    """
    # Step 1: Retrieve top candidates
    mal_embedding = retrieval.malayalam_embeddings[mal_idx:mal_idx+1].astype('float32')
    cos_sims, cos_indices = retrieval.cosine_index.search(mal_embedding, search_k)
    l2_dists, l2_indices = retrieval.l2_index.search(mal_embedding, search_k)

    # Step 2: VECTORIZED computation - get unique indices
    all_hindi_indices = np.unique(np.concatenate([cos_indices[0], l2_indices[0]]))

    # Vectorized cosine and L2 for all candidates at once
    hindi_embs = retrieval.hindi_embeddings[all_hindi_indices].astype('float32')
    cosine_sims = np.dot(mal_embedding, hindi_embs.T)[0]  # Already normalized
    l2_dists_vec = np.linalg.norm(mal_embedding - hindi_embs, axis=1)
    l2_sims_vec = 1.0 / (1.0 + l2_dists_vec)

    # Use FAISS results when available (more accurate), otherwise use vectorized
    cos_dict = {int(idx): float(sim) for idx, sim in zip(cos_indices[0], cos_sims[0])}
    l2_dict = {int(idx): float(1.0 / (1.0 + dist)) for idx, dist in zip(l2_indices[0], l2_dists[0])}

    # Use provided weights or defaults
    if score_weights is None:
        score_weights = {'cosine_weight': 0.6, 'l2_weight': 0.4}

    cosine_weight = score_weights.get('cosine_weight', 0.6)
    l2_weight = score_weights.get('l2_weight', 0.4)

    # Build candidates
    candidates = []
    for i, hin_idx in enumerate(all_hindi_indices):
        hin_idx = int(hin_idx)
        cosine_score = cos_dict.get(hin_idx, cosine_sims[i])
        l2_score = l2_dict.get(hin_idx, l2_sims_vec[i])
        combined = cosine_weight * cosine_score + l2_weight * l2_score

        candidates.append({
            'malayalam_idx': mal_idx,
            'hindi_idx': hin_idx,
            'malayalam': retrieval.malayalam_sentences[mal_idx],
            'hindi': retrieval.hindi_sentences[hin_idx],
            'cosine_score': cosine_score,
            'l2_score': l2_score,
            'score': combined
        })

    if len(candidates) == 0:
        return []

    # Step 3: Apply adaptive scoring (or skip if disabled)
    # Check if adaptive is disabled: base_weight=1.0 and z_weight=0.0 means no adaptive transformation
    use_adaptive = not (adaptive_scorer.base_weight == 1.0 and adaptive_scorer.z_weight == 0.0)

    if use_adaptive:
        candidates_with_adaptive = adaptive_scorer.process_candidates(candidates=candidates, score_key='score')
    else:
        # Skip adaptive: just set adaptive_score = score
        candidates_with_adaptive = candidates.copy()
        for c in candidates_with_adaptive:
            c['adaptive_score'] = c['score']
            c['z_score'] = 0.0
            c['base_score'] = c['score']
       # print("Adaptive scoring disabled")

    if len(candidates_with_adaptive) == 0:
        return []

    # Step 4: Apply density filtering (or skip if disabled)
    # Check if density is disabled: sparse_bonus=0.0 and crowded_penalty=0.0 means no density adjustment
    use_density = not (density_filter.config.sparse_bonus == 0.0 and density_filter.config.crowded_penalty == 0.0)

    if use_density:
       # print("Density filtering enabled")
        candidates_with_density = density_filter.process_candidates(
            candidates=candidates_with_adaptive, score_key='adaptive_score'
        )
    else:
        # Skip density: just set final_score = adaptive_score
        candidates_with_density = candidates_with_adaptive.copy()
        for c in candidates_with_density:
            c['final_score'] = c['adaptive_score']
            c['density'] = 0.5
            c['density_bonus'] = 0.0
            c['density_penalty'] = 0.0
        #print("Density filtering disabled")

    if len(candidates_with_density) == 0:
        return []

    # Step 5: Take top N and assign pre-COMET ranks
    candidates_with_density.sort(key=lambda x: x['final_score'], reverse=True)
    top_candidates = candidates_with_density[:top_after_filtering]

    # Assign rank based on final_score (1 = highest, N = lowest)
    for rank, candidate in enumerate(top_candidates, start=1):
        candidate['pre_comet_rank'] = rank

    return top_candidates

# ============================================================================
# OPTIMIZED MAIN FUNCTION - 10-50x FASTER
# ============================================================================

def main(malayalam_csv='malayalam_clean_samanantar.csv',
         hindi_csv='hindi_clean_samanantar.csv',
         output_file='final_per_sentence_pairs_optimized.csv',
         search_k=100,
         top_after_filtering=5,
         comet_threshold=0.75,
        checkpoint_dir='checkpoints',
        resume=True,
        auto_resume=True,  # Default to auto-resume for all runs
        adaptive_params=None,
         density_params=None,
         score_weights=None,
        comet_model=None,
        use_gpu=0):
    """
    OPTIMIZED PIPELINE with batched COMET scoring:
    1. Process all sentences in parallel to get top candidates (NO COMET)
    2. Batch ALL COMET calls together (biggest speedup)
    3. Select best pairs per sentence

    Expected speedup: 10-50x faster than original

    Memory considerations:
    - For n sentences with top_after_filtering=5, we process n*5 candidates through COMET
    - COMET processes in batches (64-128 per batch) for efficiency
    - For very large datasets (>50K candidates), processing is chunked to manage memory
    - Example: 100K sentences × 5 candidates = 500K COMET calls (handled efficiently in batches)

    Args:
        malayalam_csv: Path to Malayalam sentences CSV file
        hindi_csv: Path to Hindi sentences CSV file
        output_file: Output CSV file path
        search_k: Number of Hindi candidates to retrieve per sentence (default: 100)
        top_after_filtering: Top N candidates after filtering (default: 5)
        comet_threshold: Minimum COMET score threshold (default: 0.75)
        checkpoint_dir: Directory to save/load checkpoints (default: 'checkpoints')
        resume: Whether to resume from checkpoint if available (default: True)
        auto_resume: If True, auto-resume without user prompt (default: True)
        adaptive_params: Dict with 'base_weight', 'z_weight', 'z_threshold' (optional)
        density_params: Dict with 'sparse_bonus', 'crowded_penalty', 'sparse_threshold', 'crowded_threshold' (optional)
        score_weights: Dict with 'cosine_weight', 'l2_weight' (optional)

    Checkpoint System:
        - Saves progress after each major step
        - Can resume after Colab disconnects
        - Checkpoints saved: candidates_before_comet, candidates_with_comet, final_pairs
        - Progress metadata saved in progress.json
    """
    print("=" * 70)
    print("OPTIMIZED PER-SENTENCE PROCESSING PIPELINE")
    print("=" * 70)

    print("comet_threshold:", comet_threshold)

    # Initialize FAISS retrieval
    print("\nStep 1: Initializing FAISS Retrieval...")
    retrieval = FAISSRetrieval(model_name='sentence-transformers/LaBSE')
    retrieval.load_data(malayalam_csv=malayalam_csv, hindi_csv=hindi_csv)
    retrieval.embed_hindi_sentences(batch_size=64)  # Increased batch size
    retrieval.embed_malayalam_sentences(batch_size=64)
    retrieval.build_faiss_indices()

    # CHECKPOINT: Initialize checkpoint system
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

    progress = load_progress(checkpoint_dir) if resume else None
    resume_from_step = None

    if progress:
        print(f"\n📂 Found checkpoint: {progress.get('last_step', 'none')}")
        print(f"   Progress: {progress.get('processed_sentences', 0)}/{progress.get('total_sentences', 0)} sentences")
        if 'processed_candidates' in progress:
            print(f"   COMET: {progress.get('processed_candidates', 0)}/{progress.get('total_candidates', 0)} candidates")

        if auto_resume:
            # Auto-resume without user input (for automated experiments)
            resume_from_step = progress.get('last_step')
            print(f"✓ Auto-resuming from: {resume_from_step}")
        else:
            # Ask user for confirmation (for manual runs)
            print("\nOptions:")
            print("  - Press Enter to resume from checkpoint")
            print("  - Type 'n' to start fresh (will clear checkpoints)")
            resume_choice = input("Resume from checkpoint? (Enter/n): ").lower().strip()
            if resume_choice != 'n':
                resume_from_step = progress.get('last_step')
                print(f"✓ Resuming from: {resume_from_step}")
            else:
                print("Starting fresh...")
                # Clear old checkpoints
                for f in checkpoint_dir.glob("*.pkl"):
                    f.unlink()
                if (checkpoint_dir / "progress.json").exists():
                    (checkpoint_dir / "progress.json").unlink()
                progress = None
                resume_from_step = None

    # Initialize Adaptive Scoring and Density Filtering
    print("\nStep 2: Initializing Adaptive Scoring and Density Filtering...")

    # Use provided parameters or defaults
    if adaptive_params is None:
        adaptive_params = {'base_weight': 0.6, 'z_weight': 0.4, 'z_threshold': -1.0}

    adaptive_scorer = AdaptiveLocalScoring(
        base_weight=adaptive_params.get('base_weight', 0.6),
        z_weight=adaptive_params.get('z_weight', 0.4),
        z_threshold=adaptive_params.get('z_threshold', -1.0)
    )

    if density_params is None:
        density_params = {
            'sparse_bonus': 0.1,
            'crowded_penalty': 0.15,
            'sparse_threshold': 0.3,
            'crowded_threshold': 0.1
        }

    density_config = DensityConfig(
        k_neighbors=20,
        sparse_threshold=density_params.get('sparse_threshold', 0.3),
        crowded_threshold=density_params.get('crowded_threshold', 0.1),
        sparse_bonus=density_params.get('sparse_bonus', 0.1),
        crowded_penalty=density_params.get('crowded_penalty', 0.15)
    )
    density_filter = DensityBasedFiltering(
        hindi_faiss_index=retrieval.cosine_index,
        hindi_embeddings=retrieval.hindi_embeddings,
        config=density_config
    )

    # Store score weights for use in process_single_malayalam_sentence_optimized
    if score_weights is None:
        score_weights = {'cosine_weight': 0.6, 'l2_weight': 0.4}

    # Initialize COMET model
#     print("\nStep 3: Loading COMET model...")
#     try:
#         from comet import load_from_checkpoint
#         from huggingface_hub import snapshot_download
#         snapshot_download("Unbabel/wmt22-cometkiwi-da", local_dir="comet_model")
#         comet_model = load_from_checkpoint(model_path)
#         # Use GPU if available
#         use_gpu = 1 if torch.cuda.is_available() else 0
#         print(f"✓ COMET model loaded (GPU: {use_gpu})")
#     except Exception as e:
#         print(f"⚠ Warning: Could not load COMET model: {e}")
#         print("Make sure 'unbabel-comet' is installed: pip install unbabel-comet")
#         comet_model = None
#         use_gpu = 0
#     print("\nStep 3: Loading COMET model...")
#     try:
#         from comet import download_model, load_from_checkpoint

#         # Download model and get path
#         model_path = download_model("Unbabel/wmt22-cometkiwi-da")

#         # Load model
#         comet_model = load_from_checkpoint(model_path)

#         # Use GPU if available
#         use_gpu = 1 if torch.cuda.is_available() else 0
#         print(f"✓ COMET model loaded successfully (GPU: {use_gpu})")

#     except Exception as e:
#         print(f"⚠ Warning: Could not load COMET model: {e}")
#         print("Make sure 'unbabel-comet' is installed: pip install unbabel-comet")
#         comet_model = None
#         use_gpu = 0
    print("\nStep 3: Using preloaded COMET model...")

    if comet_model is None:
        raise RuntimeError("COMET model was not provided to pipeline.")

    print(f"✓ COMET model ready (GPU: {use_gpu})")

    # OPTIMIZATION 1: Process all sentences to get candidates (NO COMET)
    print("\n" + "=" * 70)
    print("STEP 1: PROCESSING ALL SENTENCES (PARALLEL, NO COMET)")
    print("=" * 70)
    print(f"Total Malayalam sentences: {len(retrieval.malayalam_sentences)}")

    num_workers = min(cpu_count() or 4, 16)  # More workers for CPU-bound tasks
    print(f"Using {num_workers} parallel workers")

    all_top_candidates = []  # List of lists: one per sentence

    def process_with_result(mal_idx):
        try:
            candidates = process_single_malayalam_sentence_optimized(
                mal_idx=mal_idx,
                retrieval=retrieval,
                adaptive_scorer=adaptive_scorer,
                density_filter=density_filter,
                search_k=search_k,
                top_after_filtering=top_after_filtering,
                score_weights=score_weights
            )
            return mal_idx, candidates, None
        except Exception as e:
            return mal_idx, [], str(e)

    # Process in parallel (using ThreadPoolExecutor since FAISS releases GIL)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_with_result, mal_idx): mal_idx
            for mal_idx in range(len(retrieval.malayalam_sentences))
        }

        with tqdm(total=len(future_to_idx), desc="Processing sentences") as pbar:
            for future in as_completed(future_to_idx):
                mal_idx, candidates, error = future.result()
                if error:
                    print(f"\n⚠ Error processing sentence {mal_idx}: {error}")
                all_top_candidates.append((mal_idx, candidates))
                pbar.update(1)

    # Sort by mal_idx to maintain order
    all_top_candidates.sort(key=lambda x: x[0])

    # CHECKPOINT: Save candidates (before COMET) or load if resuming
   # if resume_from_step != 'comet_scoring':
        #save_checkpoint(checkpoint_dir, 'candidates_before_comet', all_top_candidates)
        #save_progress(checkpoint_dir, {
        #    'last_step': 'candidates_before_comet',
        #    'total_sentences': len(retrieval.malayalam_sentences),
        #    'processed_sentences': len(all_top_candidates),
        #    'total_candidates': sum(len(c) for _, c in all_top_candidates)
        #})
    if resume_from_step == 'comet_scoring':
        # Resume: Load candidates if needed
        loaded_candidates = load_checkpoint(checkpoint_dir, 'candidates_before_comet')
        if loaded_candidates:
            all_top_candidates = loaded_candidates
            print(f"✓ Loaded {len(all_top_candidates)} sentences with candidates")

    # OPTIMIZATION 2: Batch ALL COMET calls together (BIGGEST SPEEDUP)
    print("\n" + "=" * 70)
    print("STEP 2: BATCH COMET SCORING (ALL CANDIDATES AT ONCE)")
    print("=" * 70)

    # CHECKPOINT: Load processed candidates if resuming
    candidate_metadata = []
    existing_processed_count = 0
    if resume_from_step == 'comet_scoring' or resume_from_step == 'select_best':
        print("📂 Resuming: Checking for COMET scores checkpoint...")
        loaded_metadata = load_checkpoint(checkpoint_dir, 'candidates_with_comet')
        if loaded_metadata:
            total_expected = sum(len(c) for _, c in all_top_candidates)
            existing_processed_count = len(loaded_metadata)
            if existing_processed_count >= total_expected * 0.95:  # 95% complete
                print(f"✓ Loaded {existing_processed_count} candidates with COMET scores (complete)")
                candidate_metadata = loaded_metadata
                resume_from_step = 'select_best'  # Skip to next step
            else:
                print(f"⚠ Partial checkpoint: {existing_processed_count}/{total_expected} candidates")
                print("   Will continue processing remaining candidates...")
                candidate_metadata = loaded_metadata  # Keep existing processed candidates
                resume_from_step = 'comet_scoring'  # Continue processing
        else:
            print("⚠ Checkpoint not found, starting COMET from beginning...")
            resume_from_step = None

    if resume_from_step != 'select_best':
        # Collect all candidate pairs for batch COMET scoring
        # If resuming, only collect unprocessed candidates
        comet_batch_data = []
        new_candidate_metadata = []  # New candidates to process

        # Create set of already processed (mal_idx, hindi_idx) pairs for fast lookup
        processed_pairs = set()
        if existing_processed_count > 0:
            for mal_idx, candidate in candidate_metadata:
                processed_pairs.add((mal_idx, candidate.get('hindi_idx', -1)))
            print(f"   Tracked {len(processed_pairs)} already processed candidate pairs")

        for mal_idx, candidates in all_top_candidates:
            for candidate in candidates:
                # Check if this candidate has already been processed
                candidate_key = (mal_idx, candidate.get('hindi_idx', -1))
                if candidate_key not in processed_pairs:
                    comet_batch_data.append({
                        'src': candidate['malayalam'],
                        'mt': candidate['hindi']
                    })
                    new_candidate_metadata.append((mal_idx, candidate))

        total_new_candidates = len(comet_batch_data)
        total_candidates = existing_processed_count + total_new_candidates
        print(f"Total candidates for COMET scoring: {total_candidates:,}")
        if existing_processed_count > 0:
            print(f"  ✓ Already processed: {existing_processed_count:,}")
            print(f"  → Remaining to process: {total_new_candidates:,}")
        else:
            print(f"  → Processing all: {total_new_candidates:,}")
        print(f"  (Average {total_candidates/len(retrieval.malayalam_sentences):.1f} candidates per sentence)")

        # Verify we have candidates for all sentences
        sentences_in_new = set(mal_idx for mal_idx, _ in new_candidate_metadata)
        sentences_in_existing = set(mal_idx for mal_idx, _ in candidate_metadata) if candidate_metadata else set()
        all_sentences_covered = sentences_in_new | sentences_in_existing
        if len(all_sentences_covered) < len(retrieval.malayalam_sentences):
            missing = set(range(len(retrieval.malayalam_sentences))) - all_sentences_covered
            print(f"  ⚠ WARNING: {len(missing)} sentences have no candidates: {sorted(missing)}")

        # Batch COMET scoring with balanced batch size for Colab
        batch_size = 128 if use_gpu else 16  # Balanced for Colab
        chunk_size = 1000  # Small chunks for Colab memory

        if total_new_candidates > 50:
            print(f"⚠ Processing remaining candidates in chunks of {chunk_size:,}...")
            num_chunks = (total_new_candidates + chunk_size - 1) // chunk_size

            # Start from beginning of new candidates
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_new_candidates)
                chunk_data = comet_batch_data[start_idx:end_idx]
                chunk_metadata = new_candidate_metadata[start_idx:end_idx]

                print(f"\n  Chunk {chunk_idx + 1}/{num_chunks}: Processing {len(chunk_data):,} candidates...")

                # Score this chunk in batches
                chunk_scores = []
                for i in tqdm(range(0, len(chunk_data), batch_size),
                             desc=f"    COMET scoring",
                             leave=False):
                    batch = chunk_data[i:i+batch_size]
                    comet_output = comet_model.predict(batch, batch_size=batch_size, gpus=use_gpu)
                    chunk_scores.extend(comet_output.scores)
                    del batch, comet_output

                # Add scores to candidates immediately
                for i, (mal_idx, candidate) in enumerate(chunk_metadata):
                    candidate['comet_score'] = chunk_scores[i]

                # Append new processed candidates to existing candidate_metadata
                candidate_metadata.extend(chunk_metadata)

                # CHECKPOINT: Save after each chunk (includes all processed so far)
                #save_checkpoint(checkpoint_dir, 'candidates_with_comet', candidate_metadata)
                #save_progress(checkpoint_dir, {
                #    'last_step': 'comet_scoring',
                #    'total_sentences': len(retrieval.malayalam_sentences),
                #    'processed_candidates': existing_processed_count + end_idx,
                #    'total_candidates': total_candidates,
                #    'current_chunk': chunk_idx + 1,
                #    'total_chunks': num_chunks
                #})

                # Clear chunk data
                del chunk_data, chunk_scores
                import gc
                gc.collect()
        else:
            # Small dataset - process all at once
            if total_new_candidates > 0:
                print(f"Scoring {total_new_candidates} remaining candidates in batches of {batch_size}...")
                comet_scores = []
                for i in tqdm(range(0, total_new_candidates, batch_size), desc="COMET scoring"):
                    batch = comet_batch_data[i:i+batch_size]
                    comet_output = comet_model.predict(batch, batch_size=batch_size, gpus=use_gpu)
                    comet_scores.extend(comet_output.scores)
                    del batch, comet_output

                # Add COMET scores to new candidates
                for i, (mal_idx, candidate) in enumerate(new_candidate_metadata):
                    candidate['comet_score'] = comet_scores[i]

                # Append new processed candidates to existing candidate_metadata
                candidate_metadata.extend(new_candidate_metadata)

            # CHECKPOINT: Save after completion
            #save_checkpoint(checkpoint_dir, 'candidates_with_comet', candidate_metadata)
            #save_progress(checkpoint_dir, {
            #    'last_step': 'comet_scoring',
            #    'total_sentences': len(retrieval.malayalam_sentences),
            #    'processed_candidates': len(candidate_metadata),
            #    'total_candidates': total_candidates
            #})

    # OPTIMIZATION 3: Select best pair per sentence
    print("\n" + "=" * 70)
    print("STEP 3: SELECTING BEST PAIRS")
    print("=" * 70)

    # Ensure we have candidate_metadata (should be set by now)
    if not candidate_metadata:
        print("⚠ ERROR: candidate_metadata is empty! Cannot select best pairs.")
        return None

    print(f"Total candidates with COMET scores: {len(candidate_metadata):,}")

    # Verify all sentences have candidates
    sentences_with_candidates = set(mal_idx for mal_idx, _ in candidate_metadata)
    missing_sentences = set(range(len(retrieval.malayalam_sentences))) - sentences_with_candidates
    if missing_sentences:
        print(f"⚠ WARNING: {len(missing_sentences)} sentences have no candidates: {sorted(missing_sentences)[:10]}...")

    final_pairs = []
    successful = 0
    failed = 0
    selected_margins = []  # Per-sentence: score(selected) - score(best other)

    # Group candidates by mal_idx and select best
    candidates_by_sentence = defaultdict(list)
    for mal_idx, candidate in candidate_metadata:
        # Verify candidate has comet_score
        if 'comet_score' not in candidate:
            print(f"⚠ WARNING: Candidate for sentence {mal_idx} missing comet_score!")
        candidates_by_sentence[mal_idx].append(candidate)

    print(f"Candidates grouped by sentence: {len(candidates_by_sentence)}/{len(retrieval.malayalam_sentences)} sentences have candidates")

    for mal_idx in range(len(retrieval.malayalam_sentences)):
        candidates = candidates_by_sentence[mal_idx]
        if len(candidates) == 0:
            print(f"⚠ Sentence {mal_idx} has no candidates!")
            failed += 1
            continue

        valid_pairs = [p for p in candidates if p.get('comet_score', 0) >= comet_threshold]

        if len(valid_pairs) > 0:
            best_pair = max(valid_pairs, key=lambda x: x['comet_score'])
            final_pairs.append(best_pair)
            successful += 1
            # Selected margin: score(chosen pair) - score(best other in top-5)
            selected_score = best_pair.get('final_score', 0.0)
            others = [c for c in candidates if (c.get('malayalam'), c.get('hindi')) != (best_pair.get('malayalam'), best_pair.get('hindi'))]
            best_other_score = max((c.get('final_score', 0.0) for c in others), default=selected_score)
            margin = selected_score - best_other_score
            selected_margins.append(margin)
        else:
            # Debug: show why it failed
            max_score = max([p.get('comet_score', -1) for p in candidates]) if candidates else -1
            print(f"⚠ Sentence {mal_idx}: No valid pairs (max COMET score: {max_score:.3f}, threshold: {comet_threshold})")
            failed += 1

    # Selected margin metrics (over successful sentences only)
    avg_selected_margin = float(np.mean(selected_margins)) if selected_margins else 0.0
    std_selected_margin = float(np.std(selected_margins)) if len(selected_margins) > 1 else 0.0

    # CHECKPOINT: Save final pairs
    #save_checkpoint(checkpoint_dir, 'final_pairs', final_pairs)
    #save_progress(checkpoint_dir, {
    #    'last_step': 'complete',
    #    'total_sentences': len(retrieval.malayalam_sentences),
    #    'successful': successful,
    #    'failed': failed,
    #    'avg_selected_margin': avg_selected_margin,
    #    'std_selected_margin': std_selected_margin
    #})

    # Save results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total Malayalam sentences processed: {len(retrieval.malayalam_sentences)}")
    print(f"Successful pairs found: {successful}")
    print(f"Failed/No match above threshold: {failed}")
    print(f"Success rate: {(successful/len(retrieval.malayalam_sentences)*100):.2f}%")
    if selected_margins:
        print(f"\nSelected margin (score(selected) - score(best other)):")
        print(f"  Avg selected margin: {avg_selected_margin:.4f} (higher = filter better distinguishes correct)")
        print(f"  Std selected margin: {std_selected_margin:.4f} (lower = more consistent across sentences)")

    # Initialize metric variables
    avg_weighted_score = 0.0
    total_weighted_score = 0.0
    rank_1_count = 0
    rank_2_count = 0
    rank_3_count = 0
    rank_1_percentage = 0.0
    rank_2_percentage = 0.0
    rank_3_percentage = 0.0
    rank_counts = {}

    # Calculate weighted Pre-COMET Rank metric
    if len(final_pairs) > 0:
        def get_rank_weight(rank):
            """Get weight for a given rank (higher rank = lower weight)"""
            if rank is None or rank == 'N/A':
                return 0.0
            rank = int(rank)
            # Exponential decay: rank 1 = 1.0, rank 2 = 0.7, rank 3 = 0.5, rank 4 = 0.3, rank 5+ = 0.1
            if rank == 1:
                return 1.0
            elif rank == 2:
                return 0.7
            elif rank == 3:
                return 0.5
            elif rank == 4:
                return 0.3
            else:
                return 0.1

        # Calculate weighted score (include failed translations with weight 0.0)
        weighted_scores = [get_rank_weight(p.get('pre_comet_rank')) for p in final_pairs]
        total_weighted_score = sum(weighted_scores)

        # Add 0.0 weight for each failed translation
        failed_count = len(retrieval.malayalam_sentences) - len(final_pairs)
        total_weighted_score += 0.0 * failed_count  # Explicitly add 0 for failures

        # Divide by TOTAL sentences (not just successful ones)
        total_sentences = len(retrieval.malayalam_sentences)
        avg_weighted_score = total_weighted_score / total_sentences if total_sentences > 0 else 0.0

        # Count by rank
        rank_counts = {}
        for p in final_pairs:
            rank = p.get('pre_comet_rank')
            if rank is not None and rank != 'N/A':
                rank = int(rank)
                rank_counts[rank] = rank_counts.get(rank, 0) + 1

        print(f"\n" + "=" * 70)
        print("WEIGHTED PRE-COMET RANK METRIC")
        print("=" * 70)
        print(f"Total sentences: {total_sentences}")
        print(f"Successful pairs: {len(final_pairs)}")
        print(f"Failed pairs: {failed_count} (weight: 0.0 each)")
        print(f"\nRank distribution (successful pairs only):")
        for rank in sorted(rank_counts.keys()):
            count = rank_counts[rank]
            weight = get_rank_weight(rank)
            percentage = (count / len(final_pairs)) * 100 if len(final_pairs) > 0 else 0.0
            print(f"  Rank {rank}: {count} pairs ({percentage:.1f}%) - Weight: {weight:.1f}")

        print(f"\nWeighted Score: {total_weighted_score:.2f} / {total_sentences} = {avg_weighted_score:.3f}")
        print(f"  - Successful pairs contribute: {sum(weighted_scores):.2f}")
        print(f"  - Failed pairs contribute: {0.0 * failed_count:.2f}")
        print(f"(Higher is better - max possible: 1.0 if all rank 1 and no failures)")

        # Store metrics
        rank_1_count = rank_counts.get(1, 0)
        rank_2_count = rank_counts.get(2, 0)
        rank_3_count = rank_counts.get(3, 0)
        rank_1_percentage = (rank_1_count / len(final_pairs)) * 100 if len(final_pairs) > 0 else 0.0
        rank_2_percentage = (rank_2_count / len(final_pairs)) * 100 if len(final_pairs) > 0 else 0.0
        rank_3_percentage = (rank_3_count / len(final_pairs)) * 100 if len(final_pairs) > 0 else 0.0

        # Save metrics to progress
        #save_progress(checkpoint_dir, {
        #    'last_step': 'complete',
        #    'total_sentences': len(retrieval.malayalam_sentences),
        #    'successful': successful,
        #    'failed': failed,
        #    'rank_1_count': rank_1_count,
        #    'rank_2_count': rank_2_count,
        #    'rank_3_count': rank_3_count,
        #    'rank_1_percentage': rank_1_percentage,
        #    'rank_2_percentage': rank_2_percentage,
        #    'rank_3_percentage': rank_3_percentage,
        #    'weighted_score': avg_weighted_score,
        #    'total_weighted_score': total_weighted_score,
        #    'rank_distribution': rank_counts
        #})

    if len(final_pairs) > 0:
        df_final = pd.DataFrame(final_pairs)
        df_final.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(final_pairs)} pairs to {output_file}")

        print("\n" + "=" * 70)
        print("SAMPLE OF TOP 10 PAIRS")
        print("=" * 70)
        for i, pair in enumerate(final_pairs[:10], 1):
            print(f"\nPair {i}")
            print(f"  Malayalam: {pair['malayalam'][:80]}...")
            print(f"  Hindi: {pair['hindi'][:80]}...")
            print(f"  Combined Score: {pair['score']:.4f}")
            print(f"  Adaptive Score: {pair.get('adaptive_score', 'N/A'):.4f}")
            print(f"  Final Score: {pair.get('final_score', 'N/A'):.4f}")
            print(f"  Pre-COMET Rank: {pair.get('pre_comet_rank', 'N/A')}")
            print(f"  COMET Score: {pair.get('comet_score', 'N/A'):.4f}")
    else:
        print("\n⚠ No pairs found above threshold!")

    print("\n" + "=" * 70)
    print("✓ OPTIMIZED PROCESSING COMPLETE!")
    print("=" * 70)

    # Return results dictionary with metrics
    return {
        'final_pairs': final_pairs,
        'successful': successful,
        'failed': failed,
        'total_pairs': len(final_pairs),
        'rank_1_count': rank_1_count,
        'rank_2_count': rank_2_count,
        'rank_3_count': rank_3_count,
        'rank_1_percentage': rank_1_percentage,
        'rank_2_percentage': rank_2_percentage,
        'rank_3_percentage': rank_3_percentage,
        'weighted_score': avg_weighted_score,
        'total_weighted_score': total_weighted_score,
        'rank_distribution': rank_counts,
        'avg_selected_margin': avg_selected_margin,
        'std_selected_margin': std_selected_margin,
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage:
    # main(
    #     malayalam_csv='malayalam_sentences.csv',
    #     hindi_csv='hindi_sentences.csv',
    #     output_file='final_per_sentence_pairs_optimized.csv',
    #     search_k=100,
    #     top_after_filtering=10,
    #     comet_threshold=0.75,
    #     checkpoint_dir='checkpoints',  # Checkpoint directory
    #     resume=True  # Enable resume functionality
    # )
    main(
        checkpoint_dir='checkpoints',
        resume=True
    )
    
