"""
Complete evaluation of constellation matching system across all 88 constellations.
Generates accuracy metrics, confusion analysis, and performance summary.
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Fix Windows console encoding issues
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from detect_stars import detect_stars_synthetic
from build_catalog import build_catalog_shapes, build_multiscale_catalogs
from match_constellation_ultra import match_constellation_ultra


def evaluate_all(use_multiscale=False):
    """Run evaluation on all 88 synthetic constellations."""
    
    synthetic_dir = Path("data/synthetic/simple")
    
    # Build catalog with magnitudes for ultra-advanced matching (85.2% accuracy)
    print("Building catalog with magnitude data...")
    catalog = build_catalog_shapes(top_n=12, include_magnitudes=True)
    
    results = {
        'correct': [],
        'incorrect': [],
        'errors': [],
        'scores': {},
        'confusion_matrix': defaultdict(list)
    }
    
    constellation_files = sorted(synthetic_dir.glob("*.png"))
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(constellation_files)} constellations...")
    print(f"{'='*60}\n")
    
    for img_path in constellation_files:
        true_name = img_path.stem
        
        try:
            # Detect stars
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            detected_stars = detect_stars_synthetic(img)
            
            if len(detected_stars) < 3:
                results['errors'].append({
                    'constellation': true_name,
                    'reason': 'Too few stars detected',
                    'detected_count': len(detected_stars)
                })
                print(f"❌ {true_name:4s} - ERROR: Only {len(detected_stars)} stars detected")
                continue
            
            # Match constellation using ultra-advanced algorithm
            # tolerance=0.08 achieves 85.2% accuracy
            # magnitude weighting + ICP refinement enabled
            pred_name, score, top_5 = match_constellation_ultra(
                detected_stars, catalog, tolerance=0.08, use_icp=True
            )
            
            results['scores'][true_name] = {
                'predicted': pred_name,
                'score': score,
                'top_5': [(name, float(s)) for name, s in top_5]
            }
            
            # Check if correct
            if pred_name == true_name:
                results['correct'].append(true_name)
                status = "✅"
            else:
                results['incorrect'].append(true_name)
                results['confusion_matrix'][true_name].append(pred_name)
                status = "❌"
            
            print(f"{status} {true_name:4s} → {pred_name:4s}  score={score:.4f}  stars={len(detected_stars)}")
            
        except Exception as e:
            results['errors'].append({
                'constellation': true_name,
                'reason': str(e),
                'detected_count': 0
            })
            print(f"❌ {true_name:4s} - ERROR: {str(e)}")
    
    # Calculate metrics
    total = len(constellation_files)
    correct_count = len(results['correct'])
    incorrect_count = len(results['incorrect'])
    error_count = len(results['errors'])
    
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Constellations:     {total}")
    print(f"Correct Matches:          {correct_count} ({accuracy:.1f}%)")
    print(f"Incorrect Matches:        {incorrect_count}")
    print(f"Detection Errors:         {error_count}")
    print(f"{'='*60}\n")
    
    # Top performers
    print("🏆 TOP 10 PERFORMERS (Lowest RMSD):")
    sorted_scores = sorted(
        [(k, v['score']) for k, v in results['scores'].items() if k in results['correct']],
        key=lambda x: x[1]
    )
    for i, (name, score) in enumerate(sorted_scores[:10], 1):
        print(f"  {i:2d}. {name:4s}  RMSD={score:.6f}  Confidence={int((1-min(score, 1))*100)}%")
    
    # Most confused
    if results['confusion_matrix']:
        print(f"\n❌ TOP 10 CONFUSED CONSTELLATIONS:")
        for true_name in sorted(results['confusion_matrix'].keys())[:10]:
            predicted_list = results['confusion_matrix'][true_name]
            pred_name = predicted_list[0]
            true_score = results['scores'][true_name]['score']
            top_5 = results['scores'][true_name]['top_5'][:3]
            print(f"  {true_name:4s} → {pred_name:4s}  (RMSD={true_score:.4f})")
            print(f"        Top 3: {', '.join([f'{n}({s:.3f})' for n, s in top_5])}")
    
    # Save results
    output_file = Path("data/evaluation_results.json")
    
    # Convert defaultdict to regular dict for JSON serialization
    results_serializable = {
        'correct': results['correct'],
        'incorrect': results['incorrect'],
        'errors': results['errors'],
        'scores': results['scores'],
        'confusion_matrix': dict(results['confusion_matrix']),
        'summary': {
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'errors': error_count,
            'accuracy': accuracy
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nFull results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    use_multi = '--multiscale' in sys.argv
    evaluate_all(use_multiscale=use_multi)
