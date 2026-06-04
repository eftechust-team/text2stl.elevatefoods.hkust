# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import requests
import os
from PIL import Image, ImageFilter
import io
import base64
import zipfile
import numpy as np
from scipy import interpolate, ndimage
from skimage import measure, transform, filters
import mapbox_earcut as earcut

app = Flask(__name__, static_folder='static', static_url_path='/static')

# You need to get the API Key from: https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey
# The AK/SK provided are for account access, not direct API calls
# After logging in with your AK/SK, generate an API Key in the Ark console
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
API_KEY = os.getenv("DOUBAO_API_KEY", "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    user_prompt = data['prompt']
    prompt = f"生成一个白色背景黑色填充的简笔画风格图像，做成{user_prompt}的形状"
    
    # Ensure API key is set via environment variable
    if not API_KEY:
        return jsonify({'error': 'Missing API key. Set environment variable DOUBAO_API_KEY in Render.'})

    # Call Doubao API to generate image
    response = requests.post(DOUBAO_API_URL, json={
        'model': 'doubao-seedream-4-0-250828',
        'prompt': prompt,
        'size': '1024x1024',
        'response_format': 'b64_json',
        'watermark': False
    }, headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    })
    
    print(f"Request sent - Status: {response.status_code}")
    print(f"Full Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        if 'data' in result and len(result['data']) > 0:
            image_data = result['data'][0]['b64_json']
            return jsonify({'image': image_data})
        else:
            return jsonify({'error': f'No image data in response: {result}'})
    else:
        error_response = response.text
        print(f"ERROR - Status: {response.status_code}")
        print(f"ERROR - Details: {error_response}")
        return jsonify({'error': f'API Error {response.status_code}: {error_response}'})

@app.route('/fast_generate_stl', methods=['POST'])
def fast_generate_stl():
    """Fast STL generation - creates a proper closed hollow shell from black pixels.
    Generates top, bottom, and side walls without any filling."""
    data = request.json
    image_b64 = data['image']
    height_mm = float(data.get('height', 5.0))
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        width, height = image.size
        
        # Extract ONLY black pixels (dark areas)
        threshold = 128
        image_array = np.array(image)
        mask_array = image_array < threshold
        
        # Generate STL using grid-based algorithm (cleaner topology, faster repair)
        stl_content = generate_stl_from_grid(
            mask_array,
            width,
            height,
            z_offset=0,
            thickness=height_mm
        )
        
        # Return STL directly (no ZIP for single file)
        stl_b64 = base64.b64encode(stl_content.encode('utf-8')).decode('utf-8')
        
        return jsonify({'stl_file': stl_b64})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Fast generation error: {str(e)}'})

@app.route('/generate_stl', methods=['POST'])
def generate_stl():
    data = request.json
    layers = data['layers']  # list of base64 images, one per layer
    num_layers = int(data['num_layers'])
    
    # Enforce maximum 4 layers
    if num_layers > 4:
        return jsonify({'error': 'Maximum 4 layers allowed. Please select 1, 2, 3, or 4 layers.'}), 400
    if num_layers < 1:
        return jsonify({'error': 'Minimum 1 layer required.'}), 400
    
    heights = data.get('heights') or []
    positions = data.get('positions') or []
    dilation = int(data.get('dilation') or 2)
    aa_enabled = bool(data.get('anti_aliasing', True))
    edge_threshold = int(data.get('edge_threshold') or 50)
    if edge_threshold < 0:
        edge_threshold = 0
    if edge_threshold > 255:
        edge_threshold = 255
    aa_upsample = int(data.get('aa_upsample') or 2)  # Reduced to 2 for speed
    if aa_upsample < 1:
        aa_upsample = 1
    if aa_upsample > 2:  # Max 2x upsampling
        aa_upsample = 2
    try:
        aa_sigma = float(data.get('aa_sigma') or 0.5)  # Reduced for speed
    except Exception:
        aa_sigma = 0.5
    if aa_sigma < 0:
        aa_sigma = 0
    if aa_sigma > 2.0:
        aa_sigma = 2.0
    # Normalize heights to floats with a sane default
    height_values = []
    for i in range(num_layers):
        try:
            val = float(heights[i]) if i < len(heights) else 2.0
            if val <= 0:
                val = 2.0
        except Exception:
            val = 2.0
        height_values.append(val)
    
    stl_files = []
    z_offsets = []
    for idx in range(num_layers):
        if idx == 0:
            z_offsets.append(0.0)
            continue
        placement = positions[idx] if idx < len(positions) else "stack"
        if placement == "same":
            z_offsets.append(z_offsets[idx - 1])
        else:
            z_offsets.append(z_offsets[idx - 1] + height_values[idx - 1])
    
    # Get first layer dimensions as reference for all layers
    first_image_data = base64.b64decode(layers[0])
    first_image = Image.open(io.BytesIO(first_image_data))
    first_image = first_image.convert('L')
    reference_width, reference_height = first_image.size
    
    try:
        for layer_idx, layer_image_b64 in enumerate(layers):
            # Decode base64 image
            image_data = base64.b64decode(layer_image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Normalize all layers to reference size for consistent scaling
            if image.size != (reference_width, reference_height):
                image = image.resize((reference_width, reference_height), Image.Resampling.LANCZOS)
            
            width, height = reference_width, reference_height
            
            # Optimized smoothing - skip upsampling here since contour generation handles it
            # close_size=1 and open_size=1 disable morphological closing/opening which would
            # fill the inside of hollow/ring-shaped selections.
            if aa_enabled:
                mask = smooth_binary_mask(image, threshold=edge_threshold, blur_radius=0.8, close_size=1, open_size=1)
            else:
                mask = smooth_binary_mask(image, threshold=edge_threshold, blur_radius=0, close_size=1, open_size=1)
            
            # Convert PIL image to numpy array
            mask_array = np.array(mask) > 128
            
            # Generate STL for this layer using grid-based approach (cleaner topology, faster repair)
            z_offset = z_offsets[layer_idx]
            thickness = height_values[layer_idx]
            stl_content = generate_stl_from_grid(
                mask_array,
                width,
                height,
                z_offset,
                thickness
            )
            
            stl_files.append({
                'name': f'layer_{layer_idx + 1}.stl',
                'content': stl_content
            })
        
        # Create ZIP file containing all STL files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for stl_file in stl_files:
                zip_file.writestr(stl_file['name'], stl_file['content'])
        
        zip_buffer.seek(0)
        zip_b64 = base64.b64encode(zip_buffer.read()).decode('utf-8')
        
        # Verify we have valid STL files
        if not stl_files:
            return jsonify({'error': 'No valid geometry generated. Try adjusting parameters.'})
        
        # Check each STL for validity
        for stl_file in stl_files:
            content = stl_file['content']
            if 'facet' not in content or content.count('endfacet') == 0:
                print(f"Warning: {stl_file['name']} has no valid facets")
        
        return jsonify({'zip_file': zip_b64, 'num_layers': num_layers})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Generation error: {str(e)}'})

def point_in_polygon(point, polygon):
    """Ray-casting algorithm to test if a (y,x) point is inside a polygon."""
    py, px = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def contour_contains(outer, inner):
    """Return True if inner contour is geometrically contained within outer contour."""
    # Test a few points from the inner contour against the outer polygon
    test_pts = inner[::max(1, len(inner) // 5)][:5]
    return all(point_in_polygon(pt, outer) for pt in test_pts)


def triangulate_with_holes(outer_verts, hole_verts_list):
    """Triangulate a polygon with holes using mapbox_earcut.
    Returns (triangles, all_verts) where triangles is a list of (i,j,k) index
    triples into all_verts (the combined outer + holes vertex array).
    outer_verts: (N,2) CCW array (row=y_image, col=x_image)
    hole_verts_list: list of (M,2) CW arrays
    """
    rings = [outer_verts] + hole_verts_list
    # Build combined (x, y) array for earcut — earcut uses (x, y) coords
    # Our points are (row, col) = (y_img, x_img), so col=x, row=y
    coords = []
    ring_ends = []
    offset = 0
    for ring in rings:
        for pt in ring:
            coords.append([float(pt[1]), float(pt[0])])  # [x, y]
        offset += len(ring)
        ring_ends.append(offset)

    all_xy = np.array(coords, dtype=np.float64)
    ring_ends_np = np.array(ring_ends, dtype=np.uint32)

    try:
        triangles_flat = earcut.triangulate_float64(all_xy, ring_ends_np)
    except Exception as ex:
        print(f"  earcut error: {ex}")
        return [], np.vstack(rings)

    if len(triangles_flat) == 0 or len(triangles_flat) % 3 != 0:
        return [], np.vstack(rings)

    tris = [(int(triangles_flat[i]), int(triangles_flat[i+1]), int(triangles_flat[i+2]))
            for i in range(0, len(triangles_flat), 3)]

    # all_verts in (row, col) order to match the rest of the codebase
    all_verts = np.vstack(rings)
    return tris, all_verts


def generate_hollow_shell_stl(mask_array, width, height, z_offset, thickness):
    """Generate a proper closed hollow shell from black pixels.
    Creates top, bottom, and side wall faces without any filling."""
    
    if not np.any(mask_array):
        return "solid layer\nendsolid layer\n"
    
    stl_lines = ["solid layer\n"]
    z_top = z_offset + thickness
    # Scale to fit model within ~10mm if image is ~100 pixels; adjust by actual dimensions
    scale = 10.0 / max(width, height)
    
    try:
        # Smooth the mask slightly to avoid jagged edges
        smooth_field = filters.gaussian(mask_array.astype(float), sigma=0.5, preserve_range=True)
        smooth_field = np.clip(smooth_field, 0, 1)
        
        # Find contours at 0.5 level
        contours = measure.find_contours(smooth_field, 0.5)
        
        if not contours:
            return "solid layer\nendsolid layer\n"
        
        # Filter tiny contours - increase threshold to eliminate noise
        filtered = [c for c in contours if len(c) >= 3 and abs(polygon_area(c)) >= 20.0]
        
        if not filtered:
            return "solid layer\nendsolid layer\n"
        
        # Process each contour as an outer contour with potential holes
        filtered_sorted = sorted(filtered, key=lambda c: abs(polygon_area(c)), reverse=True)
        
        # Classify as outer or hole
        assigned_as_hole = [False] * len(filtered_sorted)
        outer_with_holes = []
        
        for i in range(len(filtered_sorted)):
            if assigned_as_hole[i]:
                continue
            holes_for_i = []
            for j in range(i + 1, len(filtered_sorted)):
                if assigned_as_hole[j]:
                    continue
                if contour_contains(filtered_sorted[i], filtered_sorted[j]):
                    assigned_as_hole[j] = True
                    holes_for_i.append(j)
            outer_with_holes.append((i, holes_for_i))
        
        height_rescaled = smooth_field.shape[0]
        width_rescaled = smooth_field.shape[1]
        
        # Build geometry for each outer contour + holes
        for outer_idx, hole_indices in outer_with_holes:
            outer_raw = filtered_sorted[outer_idx]
            outer_simplified = simplify_contour(outer_raw, epsilon=0.8)
            if len(outer_simplified) < 3:
                continue
            
            outer_verts = ensure_ccw(outer_simplified)
            
            # Remove duplicate vertices
            unique_verts = [outer_verts[0]]
            for i in range(1, len(outer_verts)):
                if np.linalg.norm(outer_verts[i] - unique_verts[-1]) > 0.3:
                    unique_verts.append(outer_verts[i])
            outer_verts = np.array(unique_verts)
            
            if len(outer_verts) < 3:
                continue
            
            # Process holes
            hole_verts_list = []
            for hi in hole_indices:
                h_raw = filtered_sorted[hi]
                h_simplified = simplify_contour(h_raw, epsilon=0.8)
                if len(h_simplified) < 3:
                    continue
                h_verts = ensure_ccw(h_simplified)[::-1]  # Make CW for holes
                u = [h_verts[0]]
                for i in range(1, len(h_verts)):
                    if np.linalg.norm(h_verts[i] - u[-1]) > 0.3:
                        u.append(h_verts[i])
                h_verts_dedup = np.array(u)
                
                if len(h_verts_dedup) >= 3:
                    hole_verts_list.append(h_verts_dedup)
            
            # Triangulate outer contour
            triangles, all_verts = triangulate_with_holes(outer_verts, hole_verts_list)
            if not triangles:
                triangles = triangulate_polygon_earclip(outer_verts)
                if not triangles:
                    continue
                all_verts = outer_verts
            
            # ---- BOTTOM FACE ----
            for tri in triangles:
                v1 = [all_verts[tri[0]][1] * scale, (height_rescaled - all_verts[tri[0]][0]) * scale, z_offset]
                v2 = [all_verts[tri[1]][1] * scale, (height_rescaled - all_verts[tri[1]][0]) * scale, z_offset]
                v3 = [all_verts[tri[2]][1] * scale, (height_rescaled - all_verts[tri[2]][0]) * scale, z_offset]
                tri_str = create_triangle(v1, v2, v3)
                if tri_str:
                    stl_lines.append(tri_str)
            
            # ---- TOP FACE (reversed winding) ----
            for tri in triangles:
                v1 = [all_verts[tri[2]][1] * scale, (height_rescaled - all_verts[tri[2]][0]) * scale, z_top]
                v2 = [all_verts[tri[1]][1] * scale, (height_rescaled - all_verts[tri[1]][0]) * scale, z_top]
                v3 = [all_verts[tri[0]][1] * scale, (height_rescaled - all_verts[tri[0]][0]) * scale, z_top]
                tri_str = create_triangle(v1, v2, v3)
                if tri_str:
                    stl_lines.append(tri_str)
            
            # ---- SIDE WALLS for OUTER contour ----
            for i in range(len(outer_verts)):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % len(outer_verts)]
                v1_b = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_offset]
                v2_b = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_offset]
                v1_t = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_top]
                v2_t = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_top]
                
                tri_str = create_triangle(v1_b, v2_b, v1_t)
                if tri_str:
                    stl_lines.append(tri_str)
                tri_str = create_triangle(v2_b, v2_t, v1_t)
                if tri_str:
                    stl_lines.append(tri_str)
            
            # ---- SIDE WALLS for HOLES (inner walls) ----
            for hv in hole_verts_list:
                for i in range(len(hv)):
                    p1 = hv[i]
                    p2 = hv[(i + 1) % len(hv)]
                    v1_b = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_offset]
                    v2_b = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_offset]
                    v1_t = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_top]
                    v2_t = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_top]
                    
                    tri_str = create_triangle(v1_b, v1_t, v2_b)
                    if tri_str:
                        stl_lines.append(tri_str)
                    tri_str = create_triangle(v2_b, v1_t, v2_t)
                    if tri_str:
                        stl_lines.append(tri_str)
    
    except Exception as e:
        print(f"Hollow shell generation error: {e}")
        import traceback
        traceback.print_exc()
        return "solid layer\nendsolid layer\n"
    
    stl_lines.append("endsolid layer\n")
    return ''.join(stl_lines)


def generate_stl_from_contours(mask_array, width, height, z_offset, thickness, aa_enabled=True, aa_upsample=6, aa_sigma=0.35):
    """Generate STL with smooth edges using optimized contour extraction.
    Preserves all disconnected regions and correctly handles hollow shapes (holes).
    """
    print(f"=== Starting STL generation: shape={mask_array.shape}, z_offset={z_offset}, thickness={thickness} ===")
    
    if not np.any(mask_array):
        return "solid layer\nendsolid layer\n"

    # ---- Detect interior holes BEFORE any smoothing ----
    # This implements the two user conditions:
    #   1. Selection covers outline + white interior → mask is already mostly filled →
    #      no significant hole → produce solid shape.
    #   2. Selection covers only the black outline → mask is a thin ring →
    #      binary_fill_holes reveals a large interior hole → preserve it after smoothing.
    filled_mask = ndimage.binary_fill_holes(mask_array)
    hole_mask = filled_mask & (~mask_array)   # enclosed interior NOT selected by user
    has_holes = np.any(hole_mask)

    # Aggressive optimization - minimal upsampling
    upsample_factor = min(2, aa_upsample) if aa_enabled else 1
    # Scale to fit model within ~10mm if image is ~100 pixels; adjust by actual dimensions
    # This ensures position and size are correct relative to image dimensions
    base_scale = 10.0 / max(width, height)  # Scale to ~10mm for a typical image
    scale = base_scale / max(1, upsample_factor)
    stl_lines = ["solid layer\n"]

    try:
        # Fast smoothing path (preserved as requested)
        if aa_enabled and upsample_factor > 1:
            # Light Gaussian with reduced sigma for speed
            smooth_field = filters.gaussian(
                mask_array.astype(float),
                sigma=min(1.0, aa_sigma),
                preserve_range=True
            )
            
            # Fast linear upsampling only
            smooth_field = transform.rescale(
                smooth_field,
                upsample_factor,
                order=1,  # Linear is fastest
                anti_aliasing=False,
                preserve_range=True
            )
            smooth_field = np.clip(smooth_field, 0, 1)

            # Re-apply hole regions to prevent Gaussian blur from filling hollow shapes.
            # Without this, a thin ring blurs into a solid disc and the hole disappears.
            if has_holes:
                hole_up = transform.rescale(
                    hole_mask.astype(float),
                    upsample_factor,
                    order=0,              # nearest-neighbour: preserve crisp hole boundary
                    anti_aliasing=False,
                    preserve_range=True
                )
                smooth_field[hole_up > 0.5] = 0.0
        else:
            # No upsampling - use mask directly; holes are already preserved
            smooth_field = mask_array.astype(float)

        height_rescaled, width_rescaled = smooth_field.shape
        print(f"  Upsampled to {width_rescaled}x{height_rescaled}, scale={scale}")

        # Find contours at 0.5 level from smooth field
        contours = measure.find_contours(smooth_field, 0.5)
        
        if not contours:
            print("  No contours found, using fallback")
            return generate_stl_from_points_fallback(mask_array, width, height, z_offset, thickness)

        # Filter tiny contours - increase threshold to eliminate noise
        filtered = [c for c in contours if len(c) >= 3 and abs(polygon_area(c)) >= 20.0]
        
        if not filtered:
            print("  No valid contours after filtering")
            return generate_stl_from_points_fallback(mask_array, width, height, z_offset, thickness)

        print(f"  Processing {len(filtered)} contours")

        # ---- Classify contours as OUTER or INNER (holes) ----
        # Sort largest-area-first so outer contours come before their holes
        filtered_sorted = sorted(filtered, key=lambda c: abs(polygon_area(c)), reverse=True)

        # A contour is a hole if it is contained within another contour
        # We build a list of (outer_contour, [hole_contours]) pairs
        assigned_as_hole = [False] * len(filtered_sorted)
        outer_with_holes = []  # list of (outer_idx, [hole_idx, ...])

        for i in range(len(filtered_sorted)):
            if assigned_as_hole[i]:
                continue
            holes_for_i = []
            for j in range(i + 1, len(filtered_sorted)):
                if assigned_as_hole[j]:
                    continue
                # Check if j is contained within i
                if contour_contains(filtered_sorted[i], filtered_sorted[j]):
                    # j could be a hole of i OR an inner island within a hole;
                    # check parity: if already an even number of ancestors contain j it's a hole
                    assigned_as_hole[j] = True
                    holes_for_i.append(j)
            outer_with_holes.append((i, holes_for_i))

        # ---- Build geometry for each outer + its holes ----
        z_top = z_offset + thickness

        for outer_idx, hole_indices in outer_with_holes:
            outer_raw = filtered_sorted[outer_idx]
            outer_simplified = simplify_contour(outer_raw, epsilon=0.8)
            if len(outer_simplified) < 3:
                continue
            outer_verts = ensure_ccw(outer_simplified)

            # Remove duplicate vertices
            unique_verts = [outer_verts[0]]
            for i in range(1, len(outer_verts)):
                if np.linalg.norm(outer_verts[i] - unique_verts[-1]) > 0.3:
                    unique_verts.append(outer_verts[i])
            outer_verts = np.array(unique_verts)
            if len(outer_verts) < 3:
                continue

            # Process holes: ensure they are CCW for earcut (earcut expects holes CCW too,
            # it handles the winding internally based on hole_indices)
            hole_verts_list = []
            for hi in hole_indices:
                h_raw = filtered_sorted[hi]
                h_simplified = simplify_contour(h_raw, epsilon=0.8)
                if len(h_simplified) < 3:
                    continue
                # earcut expects holes as CW (opposite winding to outer)
                h_verts = ensure_ccw(h_simplified)[::-1]  # make CW
                u = [h_verts[0]]
                for i in range(1, len(h_verts)):
                    if np.linalg.norm(h_verts[i] - u[-1]) > 0.3:
                        u.append(h_verts[i])
                if len(u) >= 3:
                    hole_verts_list.append(np.array(u))

            # Triangulate using mapbox_earcut (handles holes natively — no bridge hack)
            triangles, all_verts = triangulate_with_holes(outer_verts, hole_verts_list)
            if not triangles:
                # fallback: triangulate outer only (ignore holes)
                triangles_fb = triangulate_polygon_earclip(outer_verts)
                if not triangles_fb:
                    continue
                triangles = triangles_fb
                all_verts = outer_verts

            # Bottom face
            for tri in triangles:
                v1 = [all_verts[tri[0]][1] * scale, (height_rescaled - all_verts[tri[0]][0]) * scale, z_offset]
                v2 = [all_verts[tri[1]][1] * scale, (height_rescaled - all_verts[tri[1]][0]) * scale, z_offset]
                v3 = [all_verts[tri[2]][1] * scale, (height_rescaled - all_verts[tri[2]][0]) * scale, z_offset]
                tri_str = create_triangle(v1, v2, v3)
                if tri_str:
                    stl_lines.append(tri_str)

            # Top face (reversed winding)
            for tri in triangles:
                v1 = [all_verts[tri[2]][1] * scale, (height_rescaled - all_verts[tri[2]][0]) * scale, z_top]
                v2 = [all_verts[tri[1]][1] * scale, (height_rescaled - all_verts[tri[1]][0]) * scale, z_top]
                v3 = [all_verts[tri[0]][1] * scale, (height_rescaled - all_verts[tri[0]][0]) * scale, z_top]
                tri_str = create_triangle(v1, v2, v3)
                if tri_str:
                    stl_lines.append(tri_str)

            # Side faces for OUTER contour
            for i in range(len(outer_verts)):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % len(outer_verts)]
                v1_b = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_offset]
                v2_b = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_offset]
                v1_t = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_top]
                v2_t = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_top]
                tri_str = create_triangle(v1_b, v2_b, v1_t)
                if tri_str:
                    stl_lines.append(tri_str)
                tri_str = create_triangle(v2_b, v2_t, v1_t)
                if tri_str:
                    stl_lines.append(tri_str)

            # Side faces for each HOLE (inner walls — normals must face INTO the hole)
            for hv in hole_verts_list:
                for i in range(len(hv)):
                    p1 = hv[i]
                    p2 = hv[(i + 1) % len(hv)]
                    v1_b = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_offset]
                    v2_b = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_offset]
                    v1_t = [p1[1] * scale, (height_rescaled - p1[0]) * scale, z_top]
                    v2_t = [p2[1] * scale, (height_rescaled - p2[0]) * scale, z_top]
                    # hv is CW, so swap winding to get outward-facing inner-wall normals
                    tri_str = create_triangle(v1_b, v1_t, v2_b)
                    if tri_str:
                        stl_lines.append(tri_str)
                    tri_str = create_triangle(v2_b, v1_t, v2_t)
                    if tri_str:
                        stl_lines.append(tri_str)

    except Exception as e:
        print(f"STL generation error: {e}")
        import traceback
        traceback.print_exc()
        return generate_stl_from_points_fallback(mask_array, width, height, z_offset, thickness)

    if stl_lines.count('facet') == 0:
        print("  No triangles generated")
        return generate_stl_from_points_fallback(mask_array, width, height, z_offset, thickness)

    stl_lines.append("endsolid layer\n")
    print(f"  Generated {stl_lines.count('facet')} triangles")
    return ''.join(stl_lines)

def laplacian_smooth_vertices(vertices, iterations=3, factor=0.3):
    """Apply Laplacian smoothing to vertices for mesh retopology.
    Smooths vertex positions to reduce jaggedness while preserving shape.
    """
    if len(vertices) < 3:
        return vertices
    
    smoothed = np.array(vertices, dtype=float)
    n = len(smoothed)
    
    for _ in range(iterations):
        new_verts = np.copy(smoothed)
        for i in range(n):
            # Get neighbors
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # Laplacian: average of neighbors
            laplacian = (smoothed[prev_idx] + smoothed[next_idx]) / 2.0
            
            # Move vertex towards Laplacian (weighted)
            new_verts[i] = smoothed[i] + factor * (laplacian - smoothed[i])
        
        smoothed = new_verts
    
    return smoothed

def smooth_contour_fourier(contour, keep_ratio=0.08):
    """Smooth contour using Fourier low-pass filtering.
    keep_ratio controls how many low-frequency components are retained.
    """
    if len(contour) < 6:
        return contour

    pts = np.array(contour, dtype=float)

    # Ensure closed contour for smooth periodic signal
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    complex_pts = pts[:, 1] + 1j * pts[:, 0]
    n = len(complex_pts)

    # FFT and keep low-frequency components
    spectrum = np.fft.fft(complex_pts)
    keep = max(3, int(n * keep_ratio))
    filtered = np.zeros_like(spectrum)

    # Keep DC and lowest frequencies on both ends
    filtered[:keep] = spectrum[:keep]
    filtered[-keep:] = spectrum[-keep:]

    smoothed = np.fft.ifft(filtered)
    out = np.column_stack([smoothed.imag, smoothed.real])

    # Drop duplicated endpoint
    return out[:-1]

def smooth_contour_savgol(contour, window=11, polyorder=3):
    """Smooth contour using Savitzky-Golay filter.
    Preserves features while creating naturally smooth curves.
    """
    from scipy.signal import savgol_filter
    
    if len(contour) < window + 2:
        return contour
    
    pts = np.array(contour, dtype=float)
    
    # Ensure window is odd and smaller than contour
    window = min(window, len(pts))
    if window % 2 == 0:
        window -= 1
    window = max(5, window)
    
    # For closed contours, extend the data
    pts_extended = np.vstack([pts[-window//2:], pts, pts[:window//2]])
    
    # Apply Savitzky-Golay filter to each dimension
    y_smooth = savgol_filter(pts_extended[:, 0], window, polyorder, mode='wrap')
    x_smooth = savgol_filter(pts_extended[:, 1], window, polyorder, mode='wrap')
    
    # Extract the smoothed interior points
    start = window // 2
    end = start + len(pts)
    
    return np.column_stack([y_smooth[start:end], x_smooth[start:end]])

def chaikin_smooth(points, iterations=2):
    """Chaikin subdivision to smooth polygonal chains."""
    if len(points) < 3:
        return points

    pts = np.array(points, dtype=float)
    for _ in range(iterations):
        new_pts = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append(q)
            new_pts.append(r)
        pts = np.array(new_pts)
    return pts

def generate_stl_from_points_fallback(mask_array, width, height, z_offset, thickness):
    """Fallback: Generate STL with smoothed contours from binary mask.
    Processes ALL valid contours (not just the largest) to preserve multi-region selections.
    """
    # Detect holes BEFORE smoothing so they survive the Gaussian
    filled_mask = ndimage.binary_fill_holes(mask_array)
    hole_mask = filled_mask & (~mask_array)
    has_holes = np.any(hole_mask)

    # Apply smoothing to the mask before contour extraction (smoothing step preserved)
    smooth_mask = filters.gaussian(mask_array.astype(float), sigma=1.5, preserve_range=True)

    # Re-apply hole regions after smoothing to prevent them from being filled
    if has_holes:
        smooth_mask[hole_mask] = 0.0
    
    # Find contours with lower threshold
    contours = measure.find_contours(smooth_mask, 0.3)
    
    if not contours:
        # Ultimate fallback to point-based
        points_indices = np.argwhere(mask_array)
        if len(points_indices) == 0:
            return "solid layer\nendsolid layer\n"
        points = [(int(pt[1]), int(pt[0])) for pt in points_indices]
        return generate_stl_from_points(points, width, height, z_offset, thickness, 
                                         dilation=2, block_size=1, scale=0.1)
    
    # Keep ALL valid contours (not just the largest) to preserve multi-region selections
    valid_contours = [c for c in contours if len(c) >= 3 and abs(polygon_area(c)) >= 2.0]
    if not valid_contours:
        return "solid layer\nendsolid layer\n"

    scale = 0.1
    stl_lines = ["solid layer\n"]
    z_top = z_offset + thickness

    # Classify outer vs hole contours (same logic as main path)
    valid_sorted = sorted(valid_contours, key=lambda c: abs(polygon_area(c)), reverse=True)
    assigned_as_hole = [False] * len(valid_sorted)
    outer_with_holes = []
    for i in range(len(valid_sorted)):
        if assigned_as_hole[i]:
            continue
        holes_for_i = []
        for j in range(i + 1, len(valid_sorted)):
            if assigned_as_hole[j]:
                continue
            if contour_contains(valid_sorted[i], valid_sorted[j]):
                assigned_as_hole[j] = True
                holes_for_i.append(j)
        outer_with_holes.append((i, holes_for_i))

    for outer_idx, hole_indices in outer_with_holes:
        contour = simplify_contour(valid_sorted[outer_idx], epsilon=1.0)
        if len(contour) < 3:
            continue
        outer_verts = ensure_ccw(contour)

        # Process holes
        hole_verts_list = []
        for hi in hole_indices:
            h_simplified = simplify_contour(valid_sorted[hi], epsilon=1.0)
            if len(h_simplified) < 3:
                continue
            h_verts = ensure_ccw(h_simplified)[::-1]  # CW for holes
            if len(h_verts) >= 3:
                hole_verts_list.append(np.array(h_verts))

        # Triangulate using mapbox_earcut (no bridge hack)
        triangles, all_verts = triangulate_with_holes(outer_verts, hole_verts_list)
        if not triangles:
            triangles_fb = triangulate_polygon_earclip(outer_verts)
            if not triangles_fb:
                continue
            triangles = triangles_fb
            all_verts = outer_verts

        # Bottom face
        for tri in triangles:
            v1 = [all_verts[tri[0]][1] * scale, (height - all_verts[tri[0]][0]) * scale, z_offset]
            v2 = [all_verts[tri[1]][1] * scale, (height - all_verts[tri[1]][0]) * scale, z_offset]
            v3 = [all_verts[tri[2]][1] * scale, (height - all_verts[tri[2]][0]) * scale, z_offset]
            stl_lines.append(create_triangle(v1, v2, v3))

        # Top face
        for tri in triangles:
            v1 = [all_verts[tri[2]][1] * scale, (height - all_verts[tri[2]][0]) * scale, z_top]
            v2 = [all_verts[tri[1]][1] * scale, (height - all_verts[tri[1]][0]) * scale, z_top]
            v3 = [all_verts[tri[0]][1] * scale, (height - all_verts[tri[0]][0]) * scale, z_top]
            stl_lines.append(create_triangle(v1, v2, v3))

        # Side faces for outer
        for i in range(len(outer_verts)):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % len(outer_verts)]
            v1_b = [p1[1] * scale, (height - p1[0]) * scale, z_offset]
            v2_b = [p2[1] * scale, (height - p2[0]) * scale, z_offset]
            v1_t = [p1[1] * scale, (height - p1[0]) * scale, z_top]
            v2_t = [p2[1] * scale, (height - p2[0]) * scale, z_top]
            stl_lines.append(create_triangle(v1_b, v2_b, v1_t))
            stl_lines.append(create_triangle(v2_b, v2_t, v1_t))

        # Side faces for holes (inner walls)
        for hv in hole_verts_list:
            for i in range(len(hv)):
                p1 = hv[i]
                p2 = hv[(i + 1) % len(hv)]
                v1_b = [p1[1] * scale, (height - p1[0]) * scale, z_offset]
                v2_b = [p2[1] * scale, (height - p2[0]) * scale, z_offset]
                v1_t = [p1[1] * scale, (height - p1[0]) * scale, z_top]
                v2_t = [p2[1] * scale, (height - p2[0]) * scale, z_top]
                stl_lines.append(create_triangle(v1_b, v1_t, v2_b))
                stl_lines.append(create_triangle(v2_b, v1_t, v2_t))

    stl_lines.append("endsolid layer\n")
    return ''.join(stl_lines)

def simplify_contour(contour, epsilon=0.5):
    """Fast distance-based contour simplification (non-recursive).
    Iteratively removes points that are close to the line between neighbors.
    """
    if len(contour) < 4:
        return contour
    
    result = [contour[0]]
    threshold_sq = epsilon * epsilon
    
    # Simple iterative approach: keep points that are far from line to next kept point
    for i in range(1, len(contour) - 1):
        d_sq = point_to_line_distance(contour[i], result[-1], contour[-1]) ** 2
        if d_sq > threshold_sq:
            result.append(contour[i])
    
    result.append(contour[-1])
    return np.array(result) if len(result) >= 3 else contour


def generate_stl_from_grid(mask_array, width, height, z_offset, thickness):
    """Generate STL using minimal surface approach - outer boundary only.
    Creates just outer contour with top/bottom and side walls.
    Vastly simpler than grid-based (only outer surface, not per-pixel).
    """
    if not np.any(mask_array):
        return "solid layer\nendsolid layer\n"
    
    try:
        # Find contours (boundaries of black regions)
        contours = measure.find_contours(mask_array.astype(float), 0.5)
        
        if not contours:
            return "solid layer\nendsolid layer\n"
        
        # Find the largest contour (outer boundary)
        largest_contour = max(contours, key=lambda c: abs(polygon_area(c)))
        
        # Very aggressive simplification to reduce triangle count dramatically
        # epsilon=2.0 removes 90%+ of points while preserving overall shape
        simplified = simplify_contour(largest_contour, epsilon=2.0)
        
        if len(simplified) < 3:
            return "solid layer\nendsolid layer\n"
        
        # Scale for STL
        scale = 10.0 / max(width, height)
        z_top = z_offset + thickness
        stl_lines = ["solid layer\n"]
        
        # Convert contour points to vertices in mm space
        # Input is (row, col) = (y_img, x_img), output needs (x_mm, y_mm)
        verts = []
        for pt in simplified:
            x_mm = pt[1] * scale          # col -> x
            y_mm = (height - pt[0]) * scale  # row -> y (inverted)
            verts.append(np.array([x_mm, y_mm]))
        
        verts = np.array(verts)
        
        if len(verts) < 3:
            return "solid layer\nendsolid layer\n"
        
        # Triangulate the polygon using earcut
        try:
            # Prepare coordinates in the format earcut expects: [[x,y], [x,y], ...]
            coords = np.array([[float(v[0]), float(v[1])] for v in verts], dtype=np.float64)
            ring_ends = np.array([len(verts)], dtype=np.uint32)
            
            # Triangulate
            triangles_flat = earcut.triangulate_float64(coords, ring_ends)
            
            # Convert flat index array to triangle tuples
            triangles = []
            for i in range(0, len(triangles_flat), 3):
                if i + 2 < len(triangles_flat):
                    triangles.append((triangles_flat[i], triangles_flat[i+1], triangles_flat[i+2]))
        except Exception as e:
            print(f"Earcut failed: {e}, using fan triangulation")
            # Fallback: simple fan triangulation from first vertex
            triangles = []
            for i in range(1, len(verts) - 1):
                triangles.append((0, i, i + 1))
        
        if not triangles:
            return "solid layer\nendsolid layer\n"
        
        # Create bottom face
        for i0, i1, i2 in triangles:
            v1 = [verts[i0][0], verts[i0][1], z_offset]
            v2 = [verts[i1][0], verts[i1][1], z_offset]
            v3 = [verts[i2][0], verts[i2][1], z_offset]
            tri_str = create_triangle(v1, v2, v3)
            if tri_str:
                stl_lines.append(tri_str)
        
        # Create top face (reverse winding)
        for i0, i1, i2 in triangles:
            v1 = [verts[i2][0], verts[i2][1], z_top]
            v2 = [verts[i1][0], verts[i1][1], z_top]
            v3 = [verts[i0][0], verts[i0][1], z_top]
            tri_str = create_triangle(v1, v2, v3)
            if tri_str:
                stl_lines.append(tri_str)
        
        # Create side wall perimeter
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            
            # Bottom edge vertices
            v1_b = [p1[0], p1[1], z_offset]
            v2_b = [p2[0], p2[1], z_offset]
            # Top edge vertices
            v1_t = [p1[0], p1[1], z_top]
            v2_t = [p2[0], p2[1], z_top]
            
            # Two triangles for the wall segment
            tri_str = create_triangle(v1_b, v2_b, v1_t)
            if tri_str:
                stl_lines.append(tri_str)
            
            tri_str = create_triangle(v2_b, v2_t, v1_t)
            if tri_str:
                stl_lines.append(tri_str)
        
        stl_lines.append("endsolid layer\n")
        return ''.join(stl_lines)
    
    except Exception as e:
        print(f"Error in generate_stl_from_grid: {e}")
        import traceback
        traceback.print_exc()
        return "solid layer\nendsolid layer\n"


def remove_colinear_points(points, angle_threshold=0.01):
    """Remove nearly-colinear points from a contour to reduce degenerate triangles.
    Points are removed if the angle at that vertex is very small (nearly straight).
    """
    if len(points) < 4:
        return points
    
    points = np.array(points)
    filtered = [points[0]]
    n = len(points)
    
    for i in range(1, n - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        
        # Vectors from p1 to neighbors
        v1 = p0 - p1
        v2 = p2 - p1
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            continue  # Skip if vectors are too small
        
        # Normalized dot product (cosine of angle)
        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # If angle is close to 180 degrees (nearly colinear), skip this point
        if abs(cos_angle + 1.0) < angle_threshold:  # Close to 180 degrees
            continue
        
        filtered.append(p1)
    
    # Add last point
    if len(points) > 0:
        filtered.append(points[-1])
    
    return np.array(filtered) if len(filtered) > 2 else points


def point_to_line_distance(point, line_start, line_end):
    """Distance from point to line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def smooth_contour_spline(contour, smoothing=0.005):
    """Smooth contour using B-spline interpolation with light smoothing to preserve shape."""
    if len(contour) < 4:
        return contour
    
    try:
        # Close the contour for seamless interpolation
        contour_closed = np.vstack([contour, contour[0:1]])
        
        # Parameterize by distance along contour
        distances = np.cumsum(np.sqrt(np.sum(np.diff(contour_closed, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        
        # Spline fit with light smoothing to keep fine details
        tck_y = interpolate.splrep(distances, contour_closed[:, 0], s=smoothing, k=min(3, len(contour_closed)-1))
        tck_x = interpolate.splrep(distances, contour_closed[:, 1], s=smoothing, k=min(3, len(contour_closed)-1))
        
        # Evaluate spline at high resolution for smooth edges
        num_points = min(len(contour) * 4, 600)  # Higher resolution for smooth curves
        distances_smooth = np.linspace(0, distances[-1], num_points)
        y_smooth = interpolate.splev(distances_smooth, tck_y)
        x_smooth = interpolate.splev(distances_smooth, tck_x)
        
        return np.column_stack([y_smooth[:-1], x_smooth[:-1]])  # Remove duplicated endpoint
    except:
        return contour

def triangulate_polygon(vertices):
    """Simple triangulation using fan triangulation from first vertex."""
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append([0, i, i + 1])
    return triangles

def polygon_area(points):
    """Signed polygon area for (y, x) points."""
    area = 0.0
    n = len(points)
    for i in range(n):
        y1, x1 = points[i]
        y2, x2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area * 0.5

def ensure_ccw(points):
    """Ensure points are in counter-clockwise order."""
    if polygon_area(points) < 0:
        return points[::-1]
    return points

def triangulate_polygon_earclip(vertices):
    """Ear clipping triangulation for simple (possibly concave) polygons.
    Returns list of index triples.
    """
    n = len(vertices)
    if n < 3:
        return []

    indices = list(range(n))
    triangles = []

    def is_convex(a, b, c):
        ay, ax = vertices[a]
        by, bx = vertices[b]
        cy, cx = vertices[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) >= 0

    def point_in_triangle(p, a, b, c):
        py, px = vertices[p]
        ay, ax = vertices[a]
        by, bx = vertices[b]
        cy, cx = vertices[c]

        v0x, v0y = cx - ax, cy - ay
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = px - ax, py - ay

        dot00 = v0x * v0x + v0y * v0y
        dot01 = v0x * v1x + v0y * v1y
        dot02 = v0x * v2x + v0y * v2y
        dot11 = v1x * v1x + v1y * v1y
        dot12 = v1x * v2x + v1y * v2y

        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    guard = 0
    while len(indices) > 2 and guard < 10000:
        guard += 1
        ear_found = False
        for i in range(len(indices)):
            prev_i = indices[(i - 1) % len(indices)]
            curr_i = indices[i]
            next_i = indices[(i + 1) % len(indices)]

            if not is_convex(prev_i, curr_i, next_i):
                continue

            is_ear = True
            for other in indices:
                if other in (prev_i, curr_i, next_i):
                    continue
                if point_in_triangle(other, prev_i, curr_i, next_i):
                    is_ear = False
                    break

            if is_ear:
                triangles.append([prev_i, curr_i, next_i])
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            break

    return triangles

def generate_stl_from_points(points, width, height, z_offset, thickness, dilation, block_size=1, scale=0.1):
    """Generate STL from 2D points as a manifold mesh with reduced triangles.
    - Removes isolated pixels
    - Emits only exterior faces
    """
    block = max(1, int(block_size))

    solid_name = "layer"
    stl_lines = ["solid layer\n"]

    # Filter out isolated pixels to smooth edges
    point_set = set(points)
    filtered_points = set()
    for x, y in points:
        has_neighbor = False
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            if (x+dx, y+dy) in point_set:
                has_neighbor = True
                break
        if has_neighbor or len(points) < 100:
            filtered_points.add((x, y))

    # Dilate points by a few pixels to close tiny gaps
    def dilate_points(src_points, radius=2):
        mask = [bytearray(width) for _ in range(height)]
        for x, y in src_points:
            if 0 <= x < width and 0 <= y < height:
                mask[y][x] = 1
        expanded = [bytearray(width) for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if mask[y][x]:
                    for dy in range(-radius, radius+1):
                        for dx in range(-radius, radius+1):
                            if dx*dx + dy*dy > radius*radius:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= nx < width and 0 <= ny < height:
                                expanded[ny][nx] = 1
        out = set()
        for y in range(height):
            row = expanded[y]
            for x in range(width):
                if row[x]:
                    out.add((x, y))
        return out

    grown_points = dilate_points(filtered_points, radius=dilation)

    # Downsample into blocks: mark block occupied if any pixel inside
    blocks = set()
    for x, y in grown_points:
        bx, by = x // block, y // block
        blocks.add((bx, by))

    size = block * scale  # block size in mm

    # Emit only exterior faces per block
    for bx, by in blocks:
        x3d = bx * block * scale
        y3d = (height - (by * block)) * scale
        z_bottom = z_offset
        z_top = z_offset + thickness

        # Bottom (always)
        stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d + size, y3d, z_bottom], [x3d + size, y3d - size, z_bottom]))
        stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d + size, y3d - size, z_bottom], [x3d, y3d - size, z_bottom]))

        # Top (always)
        stl_lines.append(create_triangle([x3d, y3d, z_top], [x3d + size, y3d - size, z_top], [x3d + size, y3d, z_top]))
        stl_lines.append(create_triangle([x3d, y3d, z_top], [x3d, y3d - size, z_top], [x3d + size, y3d - size, z_top]))

        # Helper to check neighbor
        def has_neighbor(dx, dy):
            return (bx + dx, by + dy) in blocks

        # Front (negative y in canvas after flip)
        if not has_neighbor(0, -1):
            stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d + size, y3d, z_top], [x3d + size, y3d, z_bottom]))
            stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d, y3d, z_top], [x3d + size, y3d, z_top]))

        # Back (positive y)
        if not has_neighbor(0, 1):
            stl_lines.append(create_triangle([x3d, y3d - size, z_bottom], [x3d + size, y3d - size, z_bottom], [x3d + size, y3d - size, z_top]))
            stl_lines.append(create_triangle([x3d, y3d - size, z_bottom], [x3d + size, y3d - size, z_top], [x3d, y3d - size, z_top]))

        # Left (negative x)
        if not has_neighbor(-1, 0):
            stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d, y3d - size, z_top], [x3d, y3d, z_top]))
            stl_lines.append(create_triangle([x3d, y3d, z_bottom], [x3d, y3d - size, z_bottom], [x3d, y3d - size, z_top]))

        # Right (positive x)
        if not has_neighbor(1, 0):
            stl_lines.append(create_triangle([x3d + size, y3d, z_bottom], [x3d + size, y3d, z_top], [x3d + size, y3d - size, z_top]))
            stl_lines.append(create_triangle([x3d + size, y3d, z_bottom], [x3d + size, y3d - size, z_top], [x3d + size, y3d - size, z_bottom]))

    stl_lines.append("endsolid layer\n")
    return ''.join(stl_lines)

def smooth_binary_mask(image, threshold=50, blur_radius=2.5, close_size=5, open_size=5):
    """Create a smoothed binary mask to reduce jagged edges and tiny artifacts.
    Set close_size=1 and open_size=1 to disable morphological closing/opening
    (which would fill the interior of hollow/ring-shaped selections).
    """
    # Foreground as white (255), background as black (0)
    mask = image.point(lambda p: 255 if p < threshold else 0).convert('L')

    # Close small gaps, then open to remove tiny spikes (size must be odd)
    if close_size and close_size > 1:
        # Ensure odd size
        close_size = close_size if close_size % 2 == 1 else close_size + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=close_size))
        mask = mask.filter(ImageFilter.MinFilter(size=close_size))
    if open_size and open_size > 1:
        # Ensure odd size
        open_size = open_size if open_size % 2 == 1 else open_size + 1
        mask = mask.filter(ImageFilter.MinFilter(size=open_size))
        mask = mask.filter(ImageFilter.MaxFilter(size=open_size))

    # Multi-pass Gaussian blur for smooth edges
    if blur_radius and blur_radius > 0:
        # First blur
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask = mask.point(lambda p: 255 if p >= 128 else 0)
        # Second blur for final smoothing
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius * 0.7))
        mask = mask.point(lambda p: 255 if p >= 128 else 0)

    return mask

def create_triangle(v1, v2, v3):
    """Create a triangle facet in STL ASCII format with computed normal.
    Enhanced validation for PrusaSlicer compatibility."""
    # Validate vertices are valid numbers
    try:
        for v in [v1, v2, v3]:
            if len(v) != 3:
                return ""
            for coord in v:
                # Handle numpy float types
                if isinstance(coord, (int, float, np.floating)):
                    val = float(coord)
                    if val != val or abs(val) == float('inf'):  # NaN or inf check
                        return ""
                else:
                    return ""
    except:
        return ""
    
    # Convert to float tuples with rounding for floating point precision issues
    try:
        # Round to 6 decimals to avoid floating point artifacts
        v1 = tuple(round(float(x), 6) for x in v1)
        v2 = tuple(round(float(x), 6) for x in v2)
        v3 = tuple(round(float(x), 6) for x in v3)
    except:
        return ""
    
    # Check for duplicate vertices (degenerate triangle)
    eps = 1e-7
    if (abs(v1[0] - v2[0]) < eps and abs(v1[1] - v2[1]) < eps and abs(v1[2] - v2[2]) < eps):
        return ""
    if (abs(v2[0] - v3[0]) < eps and abs(v2[1] - v3[1]) < eps and abs(v2[2] - v3[2]) < eps):
        return ""
    if (abs(v3[0] - v1[0]) < eps and abs(v3[1] - v1[1]) < eps and abs(v3[2] - v1[2]) < eps):
        return ""
    
    # Compute normal via cross product of (v2 - v1) x (v3 - v1)
    ax = v2[0] - v1[0]
    ay = v2[1] - v1[1]
    az = v2[2] - v1[2]
    bx = v3[0] - v1[0]
    by = v3[1] - v1[1]
    bz = v3[2] - v1[2]
    
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    
    # Skip degenerate triangles (zero or very small area - indicates colinearity)
    min_area_sq = 1e-12  # Minimum area squared threshold
    if length * length < min_area_sq:
        return ""
    
    # Normalize
    nx /= length
    ny /= length
    nz /= length
    
    # Validate normals
    if nx != nx or ny != ny or nz != nz or abs(nx) == float('inf') or abs(ny) == float('inf') or abs(nz) == float('inf'):
        return ""
    
    # Format with consistent precision
    return (
        f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n"
        f"    outer loop\n"
        f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n"
        f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n"
        f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n"
        f"    endloop\n"
        f"  endfacet\n"
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
