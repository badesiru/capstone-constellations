# Night Sky Constellation Detection

Automated computer vision system that detects, identifies, and labels stars and constellations in night-sky images achieving **85.2% accuracy** (75/88 correct).

## 🚀 Quick Start - Run Complete System

**Prerequisites (one-time setup):**
1. Download [HYG Database](http://www.astronexus.com/hyg) → place `hyg_v38.csv.gz` in `data/hyg/`
2. Install [Stellarium](https://stellarium.org/) to default location

Execute the full pipeline and launch interactive dashboard:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate test images (88 constellations)
python src/generate_synthetic.py

# 3. Run evaluation (achieves 85.2% accuracy)
python src/evaluate_all_constellations.py

# 4. Generate visualizations (organized by result)
python src/generate_all_visualizations.py

# 5. Launch interactive dashboard
python -m streamlit run app.py
```

**Dashboard URL:** http://localhost:8501

**What you'll see:**
- Accuracy metrics and performance charts
- Gallery of 75 correctly identified constellations
- Confusion analysis for 8 incorrect matches
- Details on 5 detection errors
- RMSD distributions and optimization timeline

**Total Time:** ~5-10 minutes for complete pipeline

---

## Complete Pipeline Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download HYG Star Database
1. Download `hyg_v38.csv.gz` from [HYG Database](http://www.astronexus.com/hyg)
2. Place in: `data/hyg/hyg_v38.csv.gz`

### Step 3: Install Stellarium (Constellation Line Data)
- Download from [stellarium.org](https://stellarium.org/)
- Default install location: `C:\Program Files\Stellarium\`
- Used for constellation line patterns and boundaries

### Step 4: Generate Synthetic Test Images
```bash
python src/generate_synthetic.py
```
**Output:** 88 constellation images in `data/synthetic/simple/`

### Step 5: Build Constellation Catalog
```bash
python src/build_catalog.py
```
**Output:** Catalog with star positions and magnitudes from Stellarium + HYG data

### Step 6: Run Full Evaluation
```bash
python src/evaluate_all_constellations.py
```
**Output:** 
- **75/88 correct (85.2% accuracy)**
- Generates `data/evaluation_results.json` with detailed metrics

### Step 7: Generate All Visualizations
```bash
python src/generate_all_visualizations.py
```
**Output:** Annotated images organized into folders:
- `data/visualizations/correct/` - 75 correctly identified constellations
- `data/visualizations/incorrect/` - 8 misidentified constellations  
- `data/visualizations/errors/` - 5 constellations with detection errors

### Step 8: Launch Interactive Dashboard
```bash
python -m streamlit run app.py
```
**Output:** Web-based UI at http://localhost:8501

**Pages:**
- **Overview** - System metrics, pie chart, top 10 performers
- **Correct Matches** - Interactive gallery with search (75 images)
- **Incorrect Matches** - Confusion matrix and misidentified pairs (8 images)
- **Detection Errors** - Error details and potential fixes (5 failures)
- **Performance Analysis** - RMSD distributions, tiers, optimization journey

### Optional: Test Single Constellation
```bash
python src/run_pipeline.py data/synthetic/simple/Ori.png --synthetic
```
**Output:** Single constellation match with annotated visualization

## Features

1. **Star Detection** - Adaptive thresholding and contour detection  
2. **Advanced Pattern Matching** - Magnitude-weighted geometric matching with ICP refinement  
3. **Geometric Features** - Triangle invariants, distance histograms, radial distribution analysis  
4. **Brightness Weighting** - Prioritizes bright stars (3x weight) for robust matching  
5. **88 Constellations** - Full IAU modern constellation set  
6. **Visualization** - Automated constellation line overlays and labels

## Results

- **75/88 correct (85.2% accuracy)**
- Techniques: Magnitude weighting + ICP refinement + ultra-tight geometric filtering

## System Requirements

**Prerequisites:**
- Python 3.8+ (Tested with 3.11, 3.12)
- [HYG Star Database v3.8](http://www.astronexus.com/hyg) - Place in `data/hyg/hyg_v38.csv.gz`
- [Stellarium](https://stellarium.org/) - Install to default location (`C:\Program Files\Stellarium\`)

**Python Packages:**
```
numpy, pandas, opencv-python, scipy, streamlit, plotly
```

**Disk Space:** ~500 MB (includes test images and visualizations)

**Estimated Time:** 5-10 minutes for complete pipeline

## Datasets

- **HYG Database v3.8** - 119,626 stars with positions and magnitudes
- **Stellarium IAU Modern** - Constellation boundaries and line connections



