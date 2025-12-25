import json
import os
import shutil
import numpy as np
import argparse
from glob import glob
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(q, t):
    """
    Convert quaternion (w, x, y, z) and translation to a 4x4 transformation matrix.
    Note: input format depends on the library. scipy uses (x, y, z, w).
    """
    # The input JSON has quaternions in an unspecified order. 
    # Let's assume standard [w, x, y, z] for now, but scipy expects [x, y, z, w]
    # If the JSON is [w, x, y, z]:
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = t
    return mat

def process_data(data_dir, output_dir, split_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, 'images')
    poses_path = os.path.join(data_dir, 'poses.json')
    
    with open(poses_path, 'r') as f:
        poses_data = json.load(f)
    
    frames = []
    
    # Check if images are panoramas (sphericalRepresentation)
    is_panorama = False
    
    for filename, pose_info in poses_data.items():
        if pose_info.get("type") == "sphericalRepresentation_jpeg":
            is_panorama = True
        
        # Construct full file path
        # Note: The JSON keys might not match exactly with files in images dir if extensions differ
        # But here they seem to include extension
        
        # Convert pose
        t = pose_info['translation']
        q = pose_info['rotation_quaternion']
        transform_matrix = quaternion_to_matrix(q, t).tolist()
        
        frames.append({
            "file_path": f"images/{filename}",
            "transform_matrix": transform_matrix
        })

    # Since we are creating a structure compatible with Nerf/3DGRUT training
    # We'll split into train/val/test
    # 3DGRUT expects transforms_train.json, transforms_val.json, transforms_test.json
    
    np.random.shuffle(frames)
    n_frames = len(frames)
    n_train = int(n_frames * split_ratio)
    n_val = int((n_frames - n_train) / 2)
    
    train_frames = frames[:n_train]
    val_frames = frames[n_train:n_train+n_val]
    test_frames = frames[n_train+n_val:]
    
    # Calculate focal length / camera angle
    # For now, we'll use a default or estimate.
    # If the images are spherical (360), 3DGRUT might need specific handling or unwrapping.
    # Standard NeRF synthetic dataset uses perspective cameras.
    # If our data is spherical, we assume it's equirectangular.
    # However, let's just output the transforms.json format first.
    
    # Assuming perspective for now or let 3DGRUT handle it if configured
    # We'll provide a dummy camera_angle_x. 
    # For 360 images, FOV is 360 horizontal.
    camera_angle_x = 0.6911112070083618 # From Lego example, approx 40 degrees?
    # Actually 2 * atan(0.5 * W / focal)
    
    # Create the output directory structure
    # We will symlink or copy the images folder
    out_images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(out_images_dir):
        # Using symlink to save space/time
        # Absolute path for symlink safety
        abs_src_images = os.path.abspath(images_dir)
        os.symlink(abs_src_images, out_images_dir)
        print(f"Created symlink for images at {out_images_dir}")

    # Write JSONs
    for split, data in zip(['train', 'val', 'test'], [train_frames, val_frames, test_frames]):
        json_content = {
            "camera_angle_x": camera_angle_x,
            "frames": data
        }
        # If spherical, might need to add specific flags if 3DGRUT supports it via json
        # Or just standard NeRF format
        
        with open(os.path.join(output_dir, f"transforms_{split}.json"), 'w') as f:
            json.dump(json_content, f, indent=4)
            
    print(f"Processed {n_frames} frames.")
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}")
    
    # Copy points.ply if exists, though 3DGRUT might generate its own or use it for init
    src_ply = os.path.join(data_dir, "points.ply")
    if os.path.exists(src_ply):
        shutil.copy2(src_ply, os.path.join(output_dir, "points.ply"))
        print("Copied points.ply")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to extracted data containing images/ and poses.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for 3DGRUT")
    args = parser.parse_args()
    
    process_data(args.data_dir, args.output_dir)