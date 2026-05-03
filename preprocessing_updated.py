#i cannot share my colab notebook since the storage ran out, so that notebook is not with saved changes
# i will send the code here, you can copy and paste that onto your colab. Make sure that your colab is pro
# since i was using one of the gpus - A100, provided by pro
# this is the code preprocessing

#to make sure colab doenst disconnect when idle
%%javascript
function ClickConnect(){
    console.log("Keeping Colab alive...");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)


#entire cell for preprocessing
from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR      = '/content/drive/MyDrive/pleural_effusion_project_UPDATED'
DRIVE_SAVE_DIR = f'{BASE_DIR}/preprocessed_20k'

for i in range(1, 11):
    os.makedirs(
        f'{DRIVE_SAVE_DIR}/preprocessed_{str(i).zfill(2)}',
        exist_ok=True
    )

# ── Preprocessing function ────────────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img   = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = img.astype(np.float32) / 255.0
    img   = np.stack([img, img, img], axis=0)
    mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img   = (img - mean[:, None, None]) / std[:, None, None]
    return img

# ── Process one image whgich is used by each parallel worker ─────────────────────
def process_one(args):
    filename, src_folder, dst_folder = args

    save_path = os.path.join(dst_folder, filename.replace('.png', '.npy'))

    # Skip if already done 
    if os.path.exists(save_path):
        return 'skipped'

    img_path = os.path.join(src_folder, filename)
    result   = preprocess_image(img_path)

    if result is None:
        return 'failed'

    np.save(save_path, result)
    return 'saved'

# ── Process all folders in parallel ──────────────────────────────────────
# NUM_WORKERS = number of images processed simultaneously
NUM_WORKERS   = 8
grand_saved   = 0
grand_skipped = 0
grand_failed  = 0

for i in range(1, 11):
    folder_name = f'images_{str(i).zfill(2)}'
    src_folder  = f'{BASE_DIR}/data/{folder_name}'
    dst_folder  = f'{DRIVE_SAVE_DIR}/preprocessed_{str(i).zfill(2)}'

    images_in_folder = [
        f for f, src in file_to_folder.items()
        if src == folder_name
    ]

    if len(images_in_folder) == 0:
        print(f"\n{folder_name} → skipping, no images needed")
        continue

    print(f"\n{'='*50}")
    print(f"Processing {folder_name} — {len(images_in_folder):,} images")
    print(f"Using {NUM_WORKERS} parallel workers")

    # Build args list for each image
    args_list = [
        (filename, src_folder, dst_folder)
        for filename in images_in_folder
    ]

    saved = skipped = failed = 0

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_one, args): args
            for args in args_list
        }
        for future in tqdm(
            as_completed(futures),
            total=len(args_list),
            desc=f"Folder {i}/10"
        ):
            result = future.result()
            if result == 'saved':
                saved += 1
            elif result == 'skipped':
                skipped += 1
            else:
                failed += 1

    grand_saved   += saved
    grand_skipped += skipped
    grand_failed  += failed

    print(f"   Saved   : {saved:,}")
    print(f"   Skipped : {skipped:,}")
    print(f"   Failed  : {failed:,}")

    # Drive storage check
    used = os.popen(f"du -sh {DRIVE_SAVE_DIR}").read().strip()
    print(f"  Drive used : {used}")

print(f"\n{'='*50}")
print(f"COMPLETE")
print(f"  Saved   : {grand_saved:,}")
print(f"  Skipped : {grand_skipped:,}")
print(f"  Failed  : {grand_failed:,}")