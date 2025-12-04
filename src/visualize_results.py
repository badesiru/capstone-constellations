
import cv2
import numpy as np
from pathlib import Path
from parse_stellarium import load_stellarium_constellations


def draw_constellation_overlay(image, detected_stars, constellation_name, score):

    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Draw detected stars as circles
    for (x, y) in detected_stars:
        cv2.circle(vis, (x, y), 5, (0, 255, 255), 2)  # Cyan circles
    
    # Try to draw constellation lines
    try:
        lines_drawn = draw_constellation_lines(vis, detected_stars, constellation_name)
    except Exception as e:
        print(f"Could not draw constellation lines: {e}")
        lines_drawn = False
    
    # Add constellation name label
    font = cv2.FONT_HERSHEY_SIMPLEX
    name_full = get_constellation_full_name(constellation_name)
    
    # Position text at top of image
    text = f"{name_full} ({constellation_name})"
    cv2.putText(vis, text, (20, 40), font, 1.2, (0, 255, 255), 2)
    
    # Add confidence score
    confidence = max(0, min(100, int(100 * (1 - min(score, 1.0)))))
    score_text = f"Confidence: {confidence}%  (RMSD: {score:.4f})"
    cv2.putText(vis, score_text, (20, 80), font, 0.7, (255, 255, 255), 2)
    
    # Add legend
    cv2.putText(vis, "Detected Stars", (20, vis.shape[0] - 60), 
                font, 0.6, (0, 255, 255), 1)
    if lines_drawn:
        cv2.putText(vis, "Constellation Lines", (20, vis.shape[0] - 35), 
                    font, 0.6, (0, 255, 0), 1)
    
    return vis


def draw_constellation_lines(image, detected_stars, constellation_name):

    # Load Stellarium constellation line data
    stell_path = r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json"
    
    if not Path(stell_path).exists():
        # Fallback: simple sequential connections if Stellarium not available
        return draw_simple_connections(image, detected_stars)
    
    try:
        import numpy as np
        from scipy.spatial.distance import cdist
        from parse_stellarium import build_catalog_from_stellarium
        from generate_synthetic import project_radec, normalize_coords
        
        constellations = load_stellarium_constellations(stell_path)
        
        if constellation_name not in constellations:
            return draw_simple_connections(image, detected_stars)
        
        const_data = constellations[constellation_name]
        lines = const_data.get("lines", [])
        
        if not lines or len(detected_stars) < 2:
            return draw_simple_connections(image, detected_stars)
        
        # Get the catalog with HIP IDs and positions
        raw_catalog = build_catalog_from_stellarium(stell_path, "data/hyg/hyg_v38.csv.gz")
        
        if constellation_name not in raw_catalog:
            return draw_simple_connections(image, detected_stars)
        
        const_stars = raw_catalog[constellation_name]
        
        # Extract HIP IDs and positions
        hip_ids = const_stars[:, 0].astype(int)
        ra = const_stars[:, 1] * 15  # Convert to degrees
        dec = const_stars[:, 2]
        mags = const_stars[:, 3]
        
        # Filter to top 12 brightest (matching our synthetic generation)
        valid = ~np.isnan(mags)
        hip_ids = hip_ids[valid]
        ra = ra[valid]
        dec = dec[valid]
        mags = mags[valid]
        
        order = np.argsort(mags)[:12]
        hip_ids = hip_ids[order]
        ra = ra[order]
        dec = dec[order]
        mags = mags[order]  # Keep magnitude info for weighting
        
        # Project catalog stars to normalized coordinates
        coords_catalog = project_radec(ra, dec)
        coords_catalog = normalize_coords(coords_catalog)
        
        # Convert detected pixel stars to normalized coordinates
        IMG_SIZE = image.shape[0]
        SCALE = int(IMG_SIZE * 0.38)  # Match generate_synthetic.py
        CX = IMG_SIZE // 2
        CY = IMG_SIZE // 2
        
        detected_normalized = []
        for px, py in detected_stars:
            x_norm = (px - CX) / SCALE
            y_norm = -(py - CY) / SCALE
            detected_normalized.append([x_norm, y_norm])
        
        detected_normalized = np.array(detected_normalized)
        
        # Match catalog stars to detected stars using advanced matching
        # Use the same approach as the ultra-advanced algorithm (85.2% accuracy)
        # Prioritize matching based on brightness and geometric proximity
        
        # Try to align the catalog with detected stars
        try:
            from scipy.spatial.distance import cdist
            
            # Compute pairwise distances
            distances = cdist(coords_catalog, detected_normalized)
            
            # For each catalog star, find closest detected star within tolerance
            # Prioritize brighter stars (lower magnitude = brighter)
            tolerance = 0.10  # Slightly looser than matching (0.08) for visualization
            catalog_to_detected = {}
            
            # Process catalog stars in order of brightness (brightest first)
            brightness_order = np.argsort(mags)
            
            for cat_idx in brightness_order:
                # Find closest detected star
                closest_det_idx = np.argmin(distances[cat_idx])
                min_dist = distances[cat_idx, closest_det_idx]
                
                if min_dist <= tolerance:
                    # Check if this detected star hasn't been claimed
                    if closest_det_idx not in catalog_to_detected.values():
                        catalog_to_detected[cat_idx] = closest_det_idx
                    else:
                        # Find who claimed it
                        for other_cat, other_det in list(catalog_to_detected.items()):
                            if other_det == closest_det_idx:
                                # Brighter stars get priority
                                if mags[cat_idx] < mags[other_cat]:
                                    # Take it from the dimmer star
                                    catalog_to_detected[cat_idx] = closest_det_idx
                                    del catalog_to_detected[other_cat]
                                break
        except:
            # Fallback to simple matching
            catalog_to_detected = {i: np.argmin(distances[i]) for i in range(len(coords_catalog))}
        
        # Create mapping from HIP ID to detected pixel position
        hip_to_pixel = {}
        for cat_idx, hip in enumerate(hip_ids):
            if cat_idx in catalog_to_detected:
                detected_idx = catalog_to_detected[cat_idx]
                hip_to_pixel[hip] = detected_stars[detected_idx]
        
        # Draw constellation lines based on Stellarium patterns
        # Only draw lines between confidently matched stars
        lines_drawn = 0
        max_line_length = IMG_SIZE * 0.4  # Sanity check: lines shouldn't span >40% of image
        
        for line_segment in lines:
            for i in range(len(line_segment) - 1):
                hip1 = line_segment[i]
                hip2 = line_segment[i + 1]
                
                if hip1 in hip_to_pixel and hip2 in hip_to_pixel:
                    pt1 = hip_to_pixel[hip1]
                    pt2 = hip_to_pixel[hip2]
                    
                    # Sanity check: reject unreasonably long lines (likely mismatches)
                    line_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    
                    if line_length < max_line_length:
                        # Draw with thickness based on star brightness
                        # Find magnitude of dimmer star in this pair
                        hip1_idx = np.where(hip_ids == hip1)[0]
                        hip2_idx = np.where(hip_ids == hip2)[0]
                        
                        thickness = 2  # Default
                        if len(hip1_idx) > 0 and len(hip2_idx) > 0:
                            max_mag = max(mags[hip1_idx[0]], mags[hip2_idx[0]])
                            # Brighter pairs get thicker lines
                            if max_mag < 2.0:
                                thickness = 3
                            elif max_mag < 3.0:
                                thickness = 2
                            else:
                                thickness = 1
                        
                        cv2.line(image, pt1, pt2, (0, 255, 0), thickness)
                        lines_drawn += 1
        
        return lines_drawn > 0
        
    except Exception as e:
        print(f"Warning: Could not load constellation lines: {e}")
        import traceback
        traceback.print_exc()
        return draw_simple_connections(image, detected_stars)


def draw_simple_connections(image, detected_stars):

    if len(detected_stars) < 2:
        return False
    
    for i in range(len(detected_stars) - 1):
        pt1 = detected_stars[i]
        pt2 = detected_stars[i + 1]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    
    # Close the pattern for many stars
    if len(detected_stars) > 3:
        cv2.line(image, detected_stars[-1], detected_stars[0], (0, 255, 0), 1)
    
    return True


def get_constellation_full_name(abbr):

    names = {
        'And': 'Andromeda', 'Ant': 'Antlia', 'Aps': 'Apus', 
        'Aql': 'Aquila', 'Aqr': 'Aquarius', 'Ara': 'Ara',
        'Ari': 'Aries', 'Aur': 'Auriga', 'Boo': 'Boötes',
        'CMa': 'Canis Major', 'CMi': 'Canis Minor', 'CVn': 'Canes Venatici',
        'Cae': 'Caelum', 'Cam': 'Camelopardalis', 'Cap': 'Capricornus',
        'Car': 'Carina', 'Cas': 'Cassiopeia', 'Cen': 'Centaurus',
        'Cep': 'Cepheus', 'Cet': 'Cetus', 'Cha': 'Chamaeleon',
        'Cir': 'Circinus', 'Cnc': 'Cancer', 'Col': 'Columba',
        'Com': 'Coma Berenices', 'CrA': 'Corona Australis', 'CrB': 'Corona Borealis',
        'Crt': 'Crater', 'Cru': 'Crux', 'Crv': 'Corvus',
        'Cyg': 'Cygnus', 'Del': 'Delphinus', 'Dor': 'Dorado',
        'Dra': 'Draco', 'Equ': 'Equuleus', 'Eri': 'Eridanus',
        'For': 'Fornax', 'Gem': 'Gemini', 'Gru': 'Grus',
        'Her': 'Hercules', 'Hor': 'Horologium', 'Hya': 'Hydra',
        'Hyi': 'Hydrus', 'Ind': 'Indus', 'LMi': 'Leo Minor',
        'Lac': 'Lacerta', 'Leo': 'Leo', 'Lep': 'Lepus',
        'Lib': 'Libra', 'Lup': 'Lupus', 'Lyn': 'Lynx',
        'Lyr': 'Lyra', 'Men': 'Mensa', 'Mic': 'Microscopium',
        'Mon': 'Monoceros', 'Mus': 'Musca', 'Nor': 'Norma',
        'Oct': 'Octans', 'Oph': 'Ophiuchus', 'Ori': 'Orion',
        'Pav': 'Pavo', 'Peg': 'Pegasus', 'Per': 'Perseus',
        'Phe': 'Phoenix', 'Pic': 'Pictor', 'PsA': 'Piscis Austrinus',
        'Psc': 'Pisces', 'Pup': 'Puppis', 'Pyx': 'Pyxis',
        'Ret': 'Reticulum', 'Scl': 'Sculptor', 'Sco': 'Scorpius',
        'Sct': 'Scutum', 'Ser': 'Serpens', 'Sex': 'Sextans',
        'Sge': 'Sagitta', 'Sgr': 'Sagittarius', 'Tau': 'Taurus',
        'Tel': 'Telescopium', 'TrA': 'Triangulum Australe', 'Tri': 'Triangulum',
        'Tuc': 'Tucana', 'UMa': 'Ursa Major', 'UMi': 'Ursa Minor',
        'Vel': 'Vela', 'Vir': 'Virgo', 'Vol': 'Volans',
        'Vul': 'Vulpecula'
    }
    return names.get(abbr, abbr)


if __name__ == "__main__":
    # Test visualization on Orion
    test_img = cv2.imread("data/synthetic/simple/Ori.png", cv2.IMREAD_GRAYSCALE)
    
    if test_img is not None:
        # Simulate detected stars (use actual detection)
        from detect_stars import detect_stars_synthetic
        stars = detect_stars_synthetic(test_img)
        
        # Create visualization
        result = draw_constellation_overlay(test_img, stars, "Ori", 0.057)
        
        # Save result
        cv2.imwrite("data/Ori_visualization_test.png", result)
        print("✓ Saved test visualization: data/Ori_visualization_test.png")
        print(f"  Detected {len(stars)} stars")
    else:
        print("Could not load test image")
