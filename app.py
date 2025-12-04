import streamlit as st
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Night Sky Constellation Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load evaluation results
@st.cache_data
def load_results():
    results_path = Path("data/evaluation_results.json")
    if not results_path.exists():
        st.error("evaluation_results.json not found! Run: python src/evaluate_all_constellations.py")
        return None
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_baseline_results():
    baseline_path = Path("data/baseline_comparison.json")
    if not baseline_path.exists():
        return None
    
    with open(baseline_path, 'r', encoding='utf-8') as f:
        return json.load(f)

results = load_results()

if results is None:
    st.stop()

# Title and Header
st.title("Night Sky Constellation Detection")
st.markdown("### Advanced Geometric Pattern Matching with 85.2% Accuracy")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select View:",
        ["Overview", "Baseline Comparison", "Correct Matches", "Incorrect Matches", "Detection Errors", "Performance Analysis"]
    )
    
    st.markdown("---")
    
    st.markdown("### Quick Stats")
    st.metric("Total", 88)
    st.metric("Correct", f"{results['summary']['correct']}")
    st.metric("Accuracy", f"{results['summary']['accuracy']:.1f}%")

# Overview Page
if page == "Overview":
    st.header("System Overview")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Constellations",
            value=results['summary']['total'],
            delta=None
        )
    
    with col2:
        st.metric(
            label="Correct",
            value=results['summary']['correct'],
            delta=f"{results['summary']['accuracy']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Incorrect",
            value=results['summary']['incorrect'],
            delta=f"{results['summary']['incorrect']/results['summary']['total']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Errors",
            value=results['summary']['errors'],
            delta=f"{results['summary']['errors']/results['summary']['total']*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Pie Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Results Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=['Correct', 'Incorrect', 'Detection Errors'],
            values=[
                results['summary']['correct'],
                results['summary']['incorrect'],
                results['summary']['errors']
            ],
            hole=0.4,
            marker=dict(colors=['#28a745', '#dc3545', '#ffc107'])
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Top 10 Performers")
        # Get top 10 by lowest RMSD from correct matches
        correct_scores = {k: v for k, v in results['scores'].items() if k in results['correct']}
        top_10 = sorted(correct_scores.items(), key=lambda x: x[1]['score'])[:10]
        
        top_df = pd.DataFrame([
            {
                'Constellation': name,
                'RMSD': data['score'],
                'Confidence': f"{(1 - min(data['score'], 1)) * 100:.1f}%"
            }
            for name, data in top_10
        ])
        st.dataframe(top_df, width='stretch', hide_index=True)
    
    st.markdown("---")

# Baseline Comparison Page
elif page == "Baseline Comparison":
    st.header("Baseline Algorithm Comparison")
    
    baseline_results = load_baseline_results()
    
    if baseline_results is None:
        st.warning("⚠️ Baseline comparison not run yet!")
        st.info("Run: `python src/evaluate_baselines.py` to generate comparison data")
        st.stop()
    
    metadata = baseline_results['metadata']
    
    # Summary metrics
    st.subheader("Accuracy based on the 83 testable constellations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Random Guessing")
        st.metric("Accuracy", f"{metadata['theoretical_random_accuracy']:.1f}%", 
                 help="Theoretical baseline: 1/88 chance")
        st.caption("Naive baseline - picks random constellation")
    
    with col2:
        st.markdown("#### Nearest-Neighbor")
        st.metric("Accuracy", f"{metadata['nearest_neighbor_accuracy']:.1f}%",
                 help="Simple centroid distance matching")
        st.caption("No geometric filtering or magnitude weighting")
    
    with col3:
        st.markdown("#### Ultra-Advanced (Ours)")
        st.metric("Accuracy", f"{metadata['ultra_advanced_accuracy']:.1f}%",
                 delta=f"+{metadata['ultra_advanced_accuracy'] - metadata['nearest_neighbor_accuracy']:.1f}pp",
                 help="Magnitude weighting + ICP + geometric filtering")
        st.caption("Current system with all optimizations")
    
    st.markdown("---")
    
    # Improvement metrics
    st.subheader("Improvement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Improvement over Random",
            f"{metadata['improvement_over_random']:.1f}x",
            help=f"{metadata['ultra_advanced_accuracy']:.1f}% / {metadata['theoretical_random_accuracy']:.1f}%"
        )
        st.caption(f"Ultra-advanced is **{metadata['improvement_over_random']:.1f}x better** than random guessing")
    
    with col2:
        st.metric(
            "Improvement over Nearest-Neighbor", 
            f"{metadata['improvement_over_nn']:.1f}x" if metadata['improvement_over_nn'] else "N/A",
            help=f"{metadata['ultra_advanced_accuracy']:.1f}% / {metadata['nearest_neighbor_accuracy']:.1f}%"
        )
        st.caption(f"Ultra-advanced is **{metadata['improvement_over_nn']:.1f}x better** than simple matching")
    
    st.markdown("---")
    
    # Technical details
    with st.expander("📊 Technical Details"):
        st.markdown("### Method Descriptions")
        
        st.markdown("**Random Guessing:**")
        st.code("prediction = random.choice(all_88_constellations)", language="python")
        st.caption("Theoretical baseline with 1/88 = 1.1% expected accuracy")
        
        st.markdown("**Nearest-Neighbor:**")
        st.code("""# Simple centroid distance
detected_centroid = mean(detected_stars)
catalog_centroid = mean(catalog_stars)
distance = euclidean(detected_centroid, catalog_centroid)
prediction = constellation_with_min_distance""", language="python")
        st.caption("Uses only centroid distance, no shape or brightness information")
        
        st.markdown("**Ultra-Advanced (Our System):**")
        st.code("""# Multi-stage advanced matching
1. Magnitude weighting (3x for bright stars)
2. Geometric triangle invariants
3. Distance histogram matching
4. Radial distribution analysis  
5. ICP refinement for alignment
6. Tight tolerance filtering (0.08)""", language="python")
        st.caption("Combines multiple sophisticated techniques for robust matching")

# Correct Matches Page
elif page == "Correct Matches":
    st.header(f"Correctly Identified Constellations ({len(results['correct'])})")
    
    # Search and filter
    search = st.text_input("Search constellation by abbreviation:", "")
    
    # Get images
    correct_dir = Path("data/visualizations/correct")
    if not correct_dir.exists():
        st.warning("⚠️ Visualization folder not found! Run: python src/generate_all_visualizations.py")
    else:
        correct_imgs = sorted(correct_dir.glob("*_annotated.jpg"))
        
        # Filter by search
        if search:
            correct_imgs = [img for img in correct_imgs if search.lower() in img.stem.lower()]
        
        # Display in grid
        cols_per_row = 3
        for i in range(0, len(correct_imgs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(correct_imgs):
                    img_path = correct_imgs[idx]
                    const_name = img_path.stem.replace("_annotated", "")
                    
                    with col:
                        st.image(str(img_path), width='stretch')
                        
                        # Show metrics
                        if const_name in results['scores']:
                            score = results['scores'][const_name]['score']
                            confidence = (1 - min(score, 1)) * 100
                            st.markdown(f"**{const_name}**")
                            st.caption(f"RMSD: {score:.6f} | Confidence: {confidence:.1f}%")

# Incorrect Matches Page
elif page == "Incorrect Matches":
    st.header(f"Incorrect Matches ({len(results['incorrect'])})")
    
    st.info("""
    These constellations were misidentified due to geometric ambiguity.
    Common causes: similar shapes, symmetric patterns, or overlapping star configurations.
    """)
    
    # Confusion analysis
    st.subheader("Confusion Matrix")
    
    confusion_data = []
    for true_const, pred_list in results['confusion_matrix'].items():
        if pred_list:
            pred_const = pred_list[0]
            score = results['scores'][true_const]['score']
            top_5 = results['scores'][true_const]['top_5']
            
            confusion_data.append({
                'True': true_const,
                'Predicted': pred_const,
                'RMSD': f"{score:.4f}",
                'Top 3 Candidates': ', '.join([f"{n}({s:.3f})" for n, s in top_5[:3]])
            })
    
    confusion_df = pd.DataFrame(confusion_data)
    st.dataframe(confusion_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Display incorrect images
    incorrect_dir = Path("data/visualizations/incorrect")
    if incorrect_dir.exists():
        incorrect_imgs = sorted(incorrect_dir.glob("*_annotated.jpg"))
        
        cols_per_row = 3
        for i in range(0, len(incorrect_imgs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(incorrect_imgs):
                    img_path = incorrect_imgs[idx]
                    const_name = img_path.stem.replace("_annotated", "")
                    
                    with col:
                        st.image(str(img_path), width='stretch')
                        
                        # Show what it was predicted as
                        if const_name in results['scores']:
                            pred = results['scores'][const_name]['predicted']
                            score = results['scores'][const_name]['score']
                            st.markdown(f"**{const_name}** → Predicted as **{pred}**")
                            st.caption(f"RMSD: {score:.4f}")

# Detection Errors Page
elif page == "Detection Errors":
    st.header(f"Detection Errors ({len(results['errors'])})")
    
    st.warning("""
    These constellations failed during star detection (< 3 stars detected).
    This is typically due to very faint stars or small constellations.
    """)
    
    # Error details table
    error_df = pd.DataFrame(results['errors'])
    st.dataframe(error_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    st.subheader("Potential Fixes")
    st.markdown("""
    1. **Adjust detection threshold** - Lower threshold to detect fainter stars
    2. **Adaptive thresholding** - Use constellation-specific thresholds
    3. **Multi-scale detection** - Try different scales for small constellations
    4. **Pre-processing enhancement** - Increase contrast for faint regions
    """)

# Performance Analysis Page
elif page == "Performance Analysis":
    st.header("Performance Analysis")
    
    # RMSD Distribution
    st.subheader("RMSD Distribution (Correct Matches Only)")
    
    correct_rmsd = [
        results['scores'][const]['score']
        for const in results['correct']
        if const in results['scores']
    ]
    
    fig = px.histogram(
        x=correct_rmsd,
        nbins=20,
        labels={'x': 'RMSD', 'y': 'Count'},
        title='Distribution of RMSD Scores'
    )
    fig.update_traces(marker_color='#28a745')
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Performance tiers
    st.subheader("Performance Tiers")
    
    tier_data = {
        'Tier': ['Excellent', 'Good', 'Fair'],
        'RMSD Range': ['< 0.1', '0.1 - 0.5', '> 0.5'],
        'Count': [
            sum(1 for r in correct_rmsd if r < 0.1),
            sum(1 for r in correct_rmsd if 0.1 <= r < 0.5),
            sum(1 for r in correct_rmsd if r >= 0.5)
        ]
    }
    tier_df = pd.DataFrame(tier_data)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(tier_df, width='stretch', hide_index=True)
    
    with col2:
        fig = px.pie(
            tier_df,
            values='Count',
            names='Tier',
            title='Performance Tier Distribution',
            color='Tier',
            color_discrete_map={
                'Excellent': '#28a745',
                'Good': '#ffc107',
                'Fair': '#dc3545'
            }
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Improvement Journey
    st.subheader("Accuracy Optimization")
    
    # Show techniques
    st.markdown("### Techniques Applied")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Stage 1: Advanced Features**
        - Triangle invariants
        - Distance histograms
        - ICP refinement
        - Radial distribution
        """)
        
        st.markdown("""
        **Stage 2: Magnitude Weighting**
        - Bright star prioritization (3x weight)
        - Magnitude-weighted RMSD
        - Magnitude centroid offset
        """)
    
    with col2:
        st.markdown("""
        **Stage 3: Ultra Filtering**
        - Tolerance reduced to 0.08
        - Aggressive geometric filtering
        - Combined all optimizations
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Night Sky Constellation Detection | Computer Vision 2025</p>
    <p>Pattern Matching Algorithm: match_constellation_ultra.py</p>
</div>
""", unsafe_allow_html=True)
