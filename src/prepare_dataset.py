import os
import glob
import shutil
import random
import numpy as np
import cv2
try:
    from src.utils import DATASET_DIR, IMG_WIDTH, IMG_HEIGHT
except ImportError:
    from utils import DATASET_DIR, IMG_WIDTH, IMG_HEIGHT

RAW_DIR = os.path.join(DATASET_DIR, 'raw')
JUW_DIR = os.path.join(RAW_DIR, 'juw_real')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
SAMPLE_LAB_DIR = os.path.join(DATASET_DIR, 'sample_lab_views')

def generate_synthetic_workstation_image(is_occupied, seed=None):
    """
    Synthesize realistic image crop of a computer lab workstation.
    is_occupied = False -> Empty seat/desk with computer monitor, keyboard, chair.
    is_occupied = True -> Occupied seat/desk with a person seated in front of PC.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Background (Lab wall/desk surface with minor lighting gradients)
    wall_color = np.random.randint(200, 235, size=3, dtype=np.uint8)
    img[:, :] = wall_color
    
    # Desk Surface (Bottom third)
    desk_top = np.random.randint(70, 95)
    desk_color = np.random.randint(110, 160, size=3, dtype=np.uint8)
    img[desk_top:, :] = desk_color
    
    # Computer Monitor (Center)
    mon_x1, mon_y1 = np.random.randint(35, 45), np.random.randint(25, 35)
    mon_x2, mon_y2 = np.random.randint(85, 95), np.random.randint(65, 75)
    
    # Monitor bezel
    cv2.rectangle(img, (mon_x1, mon_y1), (mon_x2, mon_y2), (30, 30, 30), -1)
    
    # Monitor stand
    cv2.rectangle(img, ((mon_x1 + mon_x2)//2 - 4, mon_y2), ((mon_x1 + mon_x2)//2 + 4, desk_top + 5), (40, 40, 40), -1)
    
    # Monitor screen
    if is_occupied:
        # Screen is active (bright blue/cyan or white workspace)
        screen_color = (np.random.randint(180, 240), np.random.randint(180, 240), np.random.randint(100, 220))
    else:
        # Screen is off/idle (dark gray)
        screen_color = (np.random.randint(40, 60), np.random.randint(40, 60), np.random.randint(40, 60))
    cv2.rectangle(img, (mon_x1 + 3, mon_y1 + 3), (mon_x2 - 3, mon_y2 - 3), screen_color, -1)
    
    # Keyboard on desk
    cv2.rectangle(img, (mon_x1 - 5, desk_top + 8), (mon_x2 + 5, desk_top + 18), (50, 50, 50), -1)
    
    if not is_occupied:
        # EMPTY: Empty chair visible in front of desk
        chair_color = (np.random.randint(50, 80), np.random.randint(50, 80), np.random.randint(120, 160)) # Dark blue/maroon chair
        cv2.ellipse(img, (64, 110), (28, 16), 0, 0, 180, chair_color, -1)
        # Chair backrest
        cv2.rectangle(img, (46, 85), (82, 105), chair_color, -1)
    else:
        # OCCUPIED: Head, torso, shoulders of person sitting at workstation
        skin_color = (np.random.randint(140, 200), np.random.randint(170, 220), np.random.randint(200, 250))
        cloth_color = (np.random.randint(30, 200), np.random.randint(30, 200), np.random.randint(30, 200))
        hair_color = (np.random.randint(10, 40), np.random.randint(10, 40), np.random.randint(10, 40))
        
        # Shoulders / Torso
        cv2.ellipse(img, (64, 120), (36, 25), 0, 0, 180, cloth_color, -1)
        
        # Head
        cv2.circle(img, (64, 75), 18, skin_color, -1)
        # Hair
        cv2.ellipse(img, (64, 70), (19, 14), 0, 180, 360, hair_color, -1)
        
    # Add random camera noise / subtle brightness variation
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return noisy_img

def generate_sample_lab_overview(num_seats=20, occupied_count=13, seed=42):
    """
    Generate a synthetic lab overview image containing multiple workstations
    for testing seat_counter.py and Streamlit application.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    w, h = 900, 600
    lab_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Lab floor & wall background
    lab_img[:int(h*0.25), :] = (220, 225, 230) # Wall
    lab_img[int(h*0.25):, :] = (180, 185, 190) # Floor
    
    # Determine which seats are occupied
    occupied_indices = set(random.sample(range(num_seats), occupied_count))
    
    rows, cols = 4, 5
    margin_x, margin_y = int(w * 0.05), int(h * 0.08)
    spacing_x = int((w - 2 * margin_x) / cols)
    spacing_y = int((h - 2 * margin_y) / rows)
    box_w = int(spacing_x * 0.8)
    box_h = int(spacing_y * 0.75)
    
    seat_idx = 0
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * spacing_x + int(spacing_x * 0.1)
            y = margin_y + r * spacing_y + int(spacing_y * 0.1)
            
            is_occ = seat_idx in occupied_indices
            crop = generate_synthetic_workstation_image(is_occ, seed=1000 + seat_idx)
            crop_resized = cv2.resize(crop, (box_w, box_h))
            
            lab_img[y:y+box_h, x:x+box_w] = crop_resized
            
            # Desk border line
            cv2.rectangle(lab_img, (x-2, y-2), (x+box_w+2, y+box_h+2), (100, 100, 100), 2)
            seat_idx += 1
            
    return lab_img

def setup_directories():
    """Create directory structure for dataset."""
    dirs = [
        os.path.join(RAW_DIR, 'empty'),
        os.path.join(RAW_DIR, 'occupied'),
        os.path.join(JUW_DIR, 'empty'),
        os.path.join(JUW_DIR, 'occupied'),
        os.path.join(TRAIN_DIR, 'empty'),
        os.path.join(TRAIN_DIR, 'occupied'),
        os.path.join(VAL_DIR, 'empty'),
        os.path.join(VAL_DIR, 'occupied'),
        os.path.join(TEST_DIR, 'empty'),
        os.path.join(TEST_DIR, 'occupied'),
        SAMPLE_LAB_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def populate_raw_dataset(num_samples_per_class=300):
    """Populate dataset/raw with sample synthetic workstation images if empty."""
    empty_raw_path = os.path.join(RAW_DIR, 'empty')
    occupied_raw_path = os.path.join(RAW_DIR, 'occupied')
    
    existing_empty = glob.glob(os.path.join(empty_raw_path, '*.jpg')) + glob.glob(os.path.join(empty_raw_path, '*.png'))
    existing_occ = glob.glob(os.path.join(occupied_raw_path, '*.jpg')) + glob.glob(os.path.join(occupied_raw_path, '*.png'))
    
    if len(existing_empty) < num_samples_per_class:
        print(f"[INFO] Generating {num_samples_per_class} EMPTY workstation images...")
        for i in range(num_samples_per_class):
            img = generate_synthetic_workstation_image(is_occupied=False, seed=i)
            cv2.imwrite(os.path.join(empty_raw_path, f"empty_{i+1:04d}.jpg"), img)
            
    if len(existing_occ) < num_samples_per_class:
        print(f"[INFO] Generating {num_samples_per_class} OCCUPIED workstation images...")
        for i in range(num_samples_per_class):
            img = generate_synthetic_workstation_image(is_occupied=True, seed=i + 5000)
            cv2.imwrite(os.path.join(occupied_raw_path, f"occupied_{i+1:04d}.jpg"), img)

    # Generate sample computer lab overview images for testing seat counter & app
    sample_lab_path = os.path.join(SAMPLE_LAB_DIR, 'juw_lab_sample_1.jpg')
    if not os.path.exists(sample_lab_path):
        print("[INFO] Creating sample computer lab overview image...")
        lab_img = generate_sample_lab_overview(num_seats=20, occupied_count=13, seed=42)
        cv2.imwrite(sample_lab_path, lab_img)
        
        lab_img2 = generate_sample_lab_overview(num_seats=20, occupied_count=5, seed=99)
        cv2.imwrite(os.path.join(SAMPLE_LAB_DIR, 'juw_lab_sample_2.jpg'), lab_img2)

def split_dataset(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split raw images (and JUW real images if provided) into train, validation, and test sets.
    Prevents data leakage.
    """
    print("[INFO] Splitting dataset into Train (70%), Validation (15%), and Test (15%)...")
    
    # Clear existing train, val, test splits
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in ['empty', 'occupied']:
            target_folder = os.path.join(d, cls)
            for f in glob.glob(os.path.join(target_folder, '*.*')):
                os.remove(f)
                
    for cls in ['empty', 'occupied']:
        raw_files = glob.glob(os.path.join(RAW_DIR, cls, '*.*'))
        juw_files = glob.glob(os.path.join(JUW_DIR, cls, '*.*'))
        
        all_files = raw_files + juw_files
        random.seed(42)
        random.shuffle(all_files)
        
        total = len(all_files)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        def copy_files(file_list, dest_dir):
            for f in file_list:
                fname = os.path.basename(f)
                # Prefix JUW files to distinguish in dataset
                if 'juw_real' in f:
                    fname = f"juw_{fname}"
                shutil.copy(f, os.path.join(dest_dir, cls, fname))
                
        copy_files(train_files, TRAIN_DIR)
        copy_files(val_files, VAL_DIR)
        copy_files(test_files, TEST_DIR)
        
        print(f"  Class '{cls.upper()}': Total={total} | Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")

def main():
    print("==================================================")
    print(" JUW SMART SPACE - DATASET PREPARATION PIPELINE")
    print("==================================================")
    setup_directories()
    populate_raw_dataset(num_samples_per_class=300)
    split_dataset()
    print("\n[SUCCESS] Dataset preparation complete!")

if __name__ == '__main__':
    main()
