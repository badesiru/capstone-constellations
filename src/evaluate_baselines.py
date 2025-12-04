"""
Baseline comparison to demonstrate effectiveness of advanced matching algorithm.
Compares three approaches:
1. Random guessing (theoretical baseline)
2. Simple nearest-neighbor matching (no geometric filtering)
3. Ultra-advanced matching (magnitude weighting + ICP + geometric filtering)
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import json
import random

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from detect_stars import detect_stars_synthetic
from build_catalog import build_catalog_shapes
from match_constellation_ultra import match_constellation_ultra


def random_baseline(constellation_names):
    """Baseline 1: Random guessing - theoretical baseline."""
    return random.choice(constellation_names)


def nearest_neighbor_baseline(detected_stars, catalog):
    """
    Baseline 2: Simple nearest-neighbor matching without geometric filtering.
    Uses only centroid distance, no shape matching or magnitude weighting.
    """
    if len(detected_stars) < 3:
        return None, float('inf')
    
    # Calculate centroid of detected stars
    detected_centroid = np.mean(detected_stars, axis=0)
    
    best_match = None
    best_distance = float('inf')
    
    for const_name, const_data in catalog.items():
        # Extract positions (handle both dict and array formats)
        if isinstance(const_data, dict):
            positions = const_data['coords']
        else:
            positions = const_data
        
        # Calculate catalog centroid
        catalog_centroid = np.mean(positions, axis=0)
        
        # Simple Euclidean distance between centroids
        distance = np.linalg.norm(detected_centroid - catalog_centroid)
        
        if distance < best_distance:
            best_distance = distance
            best_match = const_name
    
    return best_match, best_distance


def evaluate_all_methods():
    """Run all three methods and compare results."""
    
    synthetic_dir = Path("data/synthetic/simple")
    constellation_files = sorted(synthetic_dir.glob("*.png"))
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON EVALUATION")
    print("="*70)
    print(f"Testing on {len(constellation_files)} constellations\n")
    
    # Build catalog
    catalog = build_catalog_shapes(top_n=12, include_magnitudes=True)
    constellation_names = list(catalog.keys())
    
    results = {
        'random': {'correct': 0, 'total': 0, 'predictions': []},
        'nearest_neighbor': {'correct': 0, 'total': 0, 'predictions': []},
        'ultra_advanced': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    for i, img_path in enumerate(constellation_files, 1):
        true_name = img_path.stem
        
        # Skip if not in catalog
        if true_name not in catalog:
            continue
        
        try:
            # Detect stars
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            detected_stars = detect_stars_synthetic(img)
            
            if len(detected_stars) < 3:
                print(f"[{i:2d}/{len(constellation_files)}] {true_name} - Skipped (too few stars)")
                continue
            
            # Method 1: Random guessing
            random_pred = random_baseline(constellation_names)
            random_correct = (random_pred == true_name)
            results['random']['correct'] += random_correct
            results['random']['total'] += 1
            results['random']['predictions'].append({
                'true': true_name,
                'predicted': random_pred,
                'correct': random_correct
            })
            
            # Method 2: Nearest-neighbor
            nn_pred, nn_score = nearest_neighbor_baseline(detected_stars, catalog)
            nn_correct = (nn_pred == true_name)
            results['nearest_neighbor']['correct'] += nn_correct
            results['nearest_neighbor']['total'] += 1
            results['nearest_neighbor']['predictions'].append({
                'true': true_name,
                'predicted': nn_pred,
                'correct': nn_correct,
                'score': float(nn_score)
            })
            
            # Method 3: Ultra-advanced (current system)
            ultra_pred, ultra_score, _ = match_constellation_ultra(
                detected_stars, catalog, tolerance=0.08, use_icp=True
            )
            ultra_correct = (ultra_pred == true_name)
            results['ultra_advanced']['correct'] += ultra_correct
            results['ultra_advanced']['total'] += 1
            results['ultra_advanced']['predictions'].append({
                'true': true_name,
                'predicted': ultra_pred,
                'correct': ultra_correct,
                'score': float(ultra_score)
            })
            
            # Print progress
            status_random = "✓" if random_correct else "✗"
            status_nn = "✓" if nn_correct else "✗"
            status_ultra = "✓" if ultra_correct else "✗"
            
            print(f"[{i:2d}/{len(constellation_files)}] {true_name:5s} | "
                  f"Random:{status_random} | NN:{status_nn} | Ultra:{status_ultra}")
            
        except Exception as e:
            print(f"[{i:2d}/{len(constellation_files)}] {true_name} - Error: {e}")
            continue
    
    # Calculate accuracies
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method_name, method_results in results.items():
        total = method_results['total']
        correct = method_results['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n{method_name.replace('_', ' ').title()}:")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.1f}%")
    
    # Theoretical random baseline
    theoretical_random = 100.0 / len(constellation_names)
    print(f"\nTheoretical Random Baseline: {theoretical_random:.1f}%")
    
    # Calculate improvement factors
    nn_accuracy = (results['nearest_neighbor']['correct'] / results['nearest_neighbor']['total'] * 100)
    ultra_accuracy = (results['ultra_advanced']['correct'] / results['ultra_advanced']['total'] * 100)
    
    print(f"\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    print(f"Ultra-Advanced vs Random: {ultra_accuracy / theoretical_random:.1f}x better")
    if nn_accuracy > 0:
        print(f"Ultra-Advanced vs Nearest-Neighbor: {ultra_accuracy / nn_accuracy:.2f}x better")
        print(f"Absolute improvement: +{ultra_accuracy - nn_accuracy:.1f} percentage points")
    
    # Save results
    output_file = Path("data/baseline_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'total_constellations': len(constellation_names),
        'tested_constellations': results['ultra_advanced']['total'],
        'theoretical_random_accuracy': theoretical_random,
        'random_accuracy': (results['random']['correct'] / results['random']['total'] * 100) if results['random']['total'] > 0 else 0,
        'nearest_neighbor_accuracy': nn_accuracy,
        'ultra_advanced_accuracy': ultra_accuracy,
        'improvement_over_random': ultra_accuracy / theoretical_random,
        'improvement_over_nn': ultra_accuracy / nn_accuracy if nn_accuracy > 0 else None
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    results = evaluate_all_methods()
