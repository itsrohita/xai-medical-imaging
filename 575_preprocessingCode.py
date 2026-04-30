# To load only the image filenames from your images.txt 
with open(f'{BASE_DIR}/data/images.txt', 'r') as f:
    lines = f.readlines()

# Extracting only the image filenames 
needed_files = []
for line in lines:
    line = line.strip()
    if line == '=== UNIQUE PATIENT IDs ===':
        break                                    # stop here
    if line and line != '=== IMAGE FILENAMES ===':
        needed_files.append(line)                # adding the filename


# MAIN FUNCTION
def preprocess_image(image_path):
    """
    Takes a path to one X-ray image.
    Returns the preprocessed image as a numpy array.
    Returns None if image cannot be loaded.
    """

    # Loading as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None   

    #  Resizing to 224x224 
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # CLAHE normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Normalizing pixel values
    img = img.astype(np.float32) / 255.0

    # Converting grayscale to 3 channels 
    img = np.stack([img, img, img], axis=-1)

    # ImageNet normalization 
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std

    return img  # final shape: (224, 224, 3)


