"""
Generate annotated visualizations for ALL 88 constellations.
Automatically organizes them into correct/incorrect/error folders based on evaluation results.
"""
import sys
import json
import subprocess
from pathlib import Path
import shutil

# Fix Windows console encoding issues
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

SIMPLE_DIR = Path("data/synthetic/simple")
OUTPUT_BASE = Path("data/visualizations")
RESULTS_FILE = Path("data/evaluation_results.json")

def main():
    # Create output directories
    correct_dir = OUTPUT_BASE / "correct"
    incorrect_dir = OUTPUT_BASE / "incorrect"
    errors_dir = OUTPUT_BASE / "errors"
    
    for dir_path in [correct_dir, incorrect_dir, errors_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation results
    if not RESULTS_FILE.exists():
        print("❌ ERROR: evaluation_results.json not found!")
        print("Please run: python src/evaluate_all_constellations.py")
        return
    
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    correct_list = results['correct']
    incorrect_list = results['incorrect']
    error_list = [e['constellation'] for e in results['errors']]
    
    print("=" * 70)
    print("GENERATING VISUALIZATIONS FOR ALL 88 CONSTELLATIONS")
    print("=" * 70)
    print(f"Correct:   {len(correct_list)} → {correct_dir}")
    print(f"Incorrect: {len(incorrect_list)} → {incorrect_dir}")
    print(f"Errors:    {len(error_list)} → {errors_dir}")
    print("=" * 70)
    print()
    
    stats = {
        'correct_generated': 0,
        'incorrect_generated': 0,
        'errors_generated': 0,
        'failed': []
    }
    
    # Get all constellation files
    all_files = sorted(SIMPLE_DIR.glob("*.png"))
    total = len(all_files)
    
    for i, img_path in enumerate(all_files, 1):
        const_name = img_path.stem
        
        # Determine category
        if const_name in correct_list:
            category = "correct"
            dest_dir = correct_dir
            symbol = "✓"
        elif const_name in incorrect_list:
            category = "incorrect"
            dest_dir = incorrect_dir
            symbol = "✗"
        elif const_name in error_list:
            category = "error"
            dest_dir = errors_dir
            symbol = "⚠"
        else:
            print(f"[{i:2d}/{total}] ? {const_name} - Not in evaluation results, skipping")
            continue
        
        # Run pipeline to generate annotated image
        proc = subprocess.run(
            ["python", "src/run_pipeline.py", str(img_path), "--synthetic"],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # Check for annotated output
        annotated_src = Path("data") / f"{const_name}_annotated.jpg"
        processed_src = Path("data") / f"{const_name}_processed.jpg"
        
        if annotated_src.exists():
            # Copy both processed and annotated to appropriate folder
            annotated_dst = dest_dir / f"{const_name}_annotated.jpg"
            processed_dst = dest_dir / f"{const_name}_processed.jpg"
            
            shutil.copy(annotated_src, annotated_dst)
            if processed_src.exists():
                shutil.copy(processed_src, processed_dst)
            
            # Get prediction and score for display
            pred = "Unknown"
            score = 0.0
            for line in proc.stdout.splitlines():
                if "Predicted Constellation:" in line:
                    pred = line.split(":")[1].strip()
                if "Matching Score:" in line:
                    score = float(line.split(":")[1].strip())
            
            print(f"[{i:2d}/{total}] {symbol} {const_name:4s} → {pred:4s} | RMSD={score:.4f} | {category}")
            
            stats[f'{category}_generated'] += 1
            
            # Cleanup temp files
            annotated_src.unlink()
            if processed_src.exists():
                processed_src.unlink()
        else:
            print(f"[{i:2d}/{total}] ✗ {const_name} - Failed to generate visualization")
            stats['failed'].append(const_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Correct:   {stats['correct_generated']}/{len(correct_list)}")
    print(f"✗ Incorrect: {stats['incorrect_generated']}/{len(incorrect_list)}")
    print(f"⚠ Errors:    {stats['errors_generated']}/{len(error_list)}")
    print(f"❌ Failed:    {len(stats['failed'])}")
    print()
    print(f"All visualizations saved to: {OUTPUT_BASE}")
    print(f"  → {correct_dir}")
    print(f"  → {incorrect_dir}")
    print(f"  → {errors_dir}")
    
    if stats['failed']:
        print(f"\n❌ Failed to generate: {', '.join(stats['failed'])}")


if __name__ == "__main__":
    main()
