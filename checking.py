from google.colab import drive
drive.mount('/content/drive')

import os
import subprocess

# Check their MyDrive contents
print("=== MyDrive contents ===")
for item in sorted(os.listdir('/content/drive/MyDrive')):
    full = os.path.join('/content/drive/MyDrive', item)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        print(f"  [FOLDER] {item}/ — {count} items")
    else:
        size = os.path.getsize(full) / 1e6
        print(f"  [FILE]   {item} — {size:.1f} MB")

# Check total storage used
print("\n=== Storage ===")
result = subprocess.run(['df', '-h', '/content/drive'], 
                       capture_output=True, text=True)
print(result.stdout)

# Check specifically for segmented folder
seg_path = '/content/drive/MyDrive/pleural_effusion_project_UPDATED/segmented_20k'
if os.path.exists(seg_path):
    print(f"\n segmented_20k EXISTS")
    for i in range(1, 11):
        folder = f'{seg_path}/segmented_{str(i).zfill(2)}'
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) 
                        if f.endswith('.npy')])
            print(f"  segmented_{str(i).zfill(2)} : {count} files")
else:
    print(f"\n segmented_20k NOT found in partner's Drive")