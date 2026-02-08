import os                         
import numpy as np                
import pandas as pd               
from glob import glob             


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm


SEED = 42  

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3  

BATCH_SIZE = 8     
EPOCHS = 25       
VALIDATION_SPLIT = 0.2  
LEARNING_RATE = 1e-4   
THRESHOLD = 0.35
TRAIN_IMAGES_DIR = "train_images"
TRAIN_MASKS_DIR = "train_masks"
TEST_IMAGES_DIR = "test_images_padded"
SUBMISSION_FILE = "submission.csv"
TRAIN_MODEL = False

def load_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
  
    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    
  
    img_array = img_to_array(img)
 
    img_array = img_array / 255.0
    
    return img_array


def load_mask(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
 
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    

    mask_array = img_to_array(mask)

    mask_array = (mask_array >= 128).astype(np.float32)
    
    return mask_array


def load_training_data():
 
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    image_paths = sorted(glob(os.path.join(TRAIN_IMAGES_DIR, "*.png")))
    
    print(f"Found {len(image_paths)} training images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {TRAIN_IMAGES_DIR}! Check your folder path.")

    images = []
    masks = []

    for image_path in tqdm(image_paths, desc="Loading images"):

        filename = os.path.basename(image_path)
        

        mask_path = os.path.join(TRAIN_MASKS_DIR, filename)
        

        if not os.path.exists(mask_path):
            print(f"WARNING: Mask not found for {filename}, skipping...")
            continue

        image = load_image(image_path)
        mask = load_mask(mask_path)
        
        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    
    print(f"\nLoaded {len(images)} image-mask pairs")
    print(f"Images shape: {images.shape}")  # (N, 256, 256, 3)
    print(f"Masks shape: {masks.shape}")    # (N, 256, 256, 1)
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mask unique values: {np.unique(masks)}")  # Should be [0, 1]
    
    return images, masks


def load_test_data():
  
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    
    image_paths = sorted(glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    
    print(f"Found {len(image_paths)} test images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {TEST_IMAGES_DIR}! Check your folder path.")
    
    images = []
    filenames = []
    
    for image_path in tqdm(image_paths, desc="Loading test images"):

        filename = os.path.basename(image_path)
        
        image = load_image(image_path)
        
        images.append(image)
        filenames.append(filename)
    
    images = np.array(images, dtype=np.float32)
    
    print(f"\nLoaded {len(images)} test images")
    print(f"Test images shape: {images.shape}")
    
    return images, filenames

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Second convolution
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)

    skip = x

    pooled = layers.MaxPooling2D(pool_size=2)(x)
    
    return skip, pooled


def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, kernel_size=2, strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    

    x = conv_block(x, num_filters)
    
    return x


def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    print("\n" + "=" * 60)
    print("BUILDING U-NET MODEL")
    print("=" * 60)
    
    inputs = layers.Input(shape=input_shape)
    skip1, pool1 = encoder_block(inputs, 32)

    skip2, pool2 = encoder_block(pool1, 64)
    skip3, pool3 = encoder_block(pool2, 128)
    skip4, pool4 = encoder_block(pool3, 256)
    bottleneck = conv_block(pool4, 512)
    dec4 = decoder_block(bottleneck, skip4, 256)
    
    dec3 = decoder_block(dec4, skip3, 128)
    
    dec2 = decoder_block(dec3, skip2, 64)
    
    dec1 = decoder_block(dec2, skip1, 32)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(dec1)
    
    model = Model(inputs=inputs, outputs=outputs, name="UNet")
    
    print(f"Model created with {model.count_params():,} parameters")
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    )
    
    return dice


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce) 
    
    # Dice Loss
    dice = dice_loss(y_true, y_pred)

    return 0.3 * bce + 0.7 * dice



def iou_metric(y_true, y_pred, threshold=0.4):
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred_binary, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou

def create_callbacks():
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="best_unet_model.keras",  
            monitor="val_loss",                 
            mode="min",                        
            save_best_only=True,               
            verbose=1                          
        ),
        
        keras.callbacks.EarlyStopping(
            monitor="val_loss",        
            mode="min",                 
            patience=5,                 
            restore_best_weights=True,  
            verbose=1
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",     
            mode="min",             
            factor=0.5,             
            patience=3,             
            min_lr=1e-7,            
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("COMPILING MODEL")
    print("=" * 60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,  # Our custom combined loss
        metrics=[
            dice_coefficient,  # Track Dice during training
            iou_metric        # Track IoU during training
        ]
    )
    
    print(f"Optimizer: Adam (learning_rate={LEARNING_RATE})")
    print(f"Loss: Combined (BCE + Dice)")
    print(f"Metrics: Dice Coefficient, IoU")
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    print("-" * 60)
    
    # Get callbacks
    callbacks = create_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,           # Training data
        validation_data=(X_val, y_val),  # Validation data
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1                    # Show progress bar
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return history

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0]

    runs = []
    for i in range(0, len(changes), 2):
        start = changes[i] + 1
        length = changes[i+1] - changes[i]
        runs.extend([str(start), str(length)])

    return " ".join(runs)


def rle_encode_simple(mask):

    pixels = mask.T.flatten() 

    runs = []
    in_run = False
    start = 0
    
    for i, pixel in enumerate(pixels):
        if pixel == 1 and not in_run:
            # Starting a new run
            start = i + 1  # 1-indexed
            in_run = True
        elif pixel == 0 and in_run:
            # Ending a run
            length = i - start + 1
            runs.extend([str(start), str(length)])
            in_run = False
    
    if in_run:
        length = len(pixels) - start + 1
        runs.extend([str(start), str(length)])
    
    return " ".join(runs) if runs else ""



def predict_and_create_submission(model, test_images, test_filenames):
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    
    print("Running model inference...")
    predictions = model.predict(test_images, batch_size=BATCH_SIZE, verbose=1)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction value range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print(f"Applying threshold of {THRESHOLD}...")
    binary_masks = (predictions >= THRESHOLD).astype(np.uint8)
    
    binary_masks = binary_masks[:, :, :, 0]
    
    print(f"\nBinary masks shape: {binary_masks.shape}")
    
    print("\nEncoding masks with RLE...")
    submission_data = []
    
    for i in tqdm(range(len(test_filenames)), desc="RLE Encoding"):
        filename = test_filenames[i]
        mask = binary_masks[i]
        
        image_id = os.path.splitext(filename)[0]

        
        rle = rle_encode(mask)
        
        if not rle:
            rle = ""  # or "-1" depending on competition format
        
        submission_data.append({
            "image_id": image_id,
            "encoded_pixels": rle
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    submission_df = submission_df.sort_values("image_id")
    
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("SUBMISSION FILE CREATED")
    print("=" * 60)
    print(f"File: {SUBMISSION_FILE}")
    print(f"Number of rows: {len(submission_df)}")
    print(f"\nFirst 5 rows:")
    print(submission_df.head())
    
    # Statistics
    empty_masks = sum(1 for rle in submission_df["encoded_pixels"] if rle == "")
    print(f"\nEmpty masks: {empty_masks} / {len(submission_df)}")
    
    return SUBMISSION_FILE



def main():
    print("\n" + "=" * 60)
    print("TERRASEG: RUGGED TERRAIN SEGMENTATION")
    print("Binary Semantic Segmentation with U-Net")
    print("=" * 60)
    
    images, masks = load_training_data()
    
    
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks,
        test_size=VALIDATION_SPLIT,  # 20% for validation
        random_state=SEED            # For reproducibility
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    del images, masks
    
    model = build_unet()
    
    print("\nModel Summary:")
    model.summary()
    
    history = None
    if TRAIN_MODEL:
        history = train_model(model, X_train, y_train, X_val, y_val)
    else:
        print("Skipping training. Using saved weights.")
    print("\n" + "=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)
    
    if os.path.exists("best_unet_model.keras"):
        print("Loading best model from checkpoint...")
        model = keras.models.load_model(
            "best_unet_model.keras",
            custom_objects={
                "combined_loss": combined_loss,
                "dice_coefficient": dice_coefficient,
                "iou_metric": iou_metric
            }
        )
        print("Best model loaded successfully!")
    else:
        print("No checkpoint found, using final model from training.")
    
    test_images, test_filenames = load_test_data()
    
    submission_file = predict_and_create_submission(model, test_images, test_filenames)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    if history is not None:
        print(f"""
    Summary:
    --------
    - Model trained for {len(history.history['loss'])} epochs
    - Best validation loss: {min(history.history['val_loss']):.4f}
    - Best validation Dice: {max(history.history['val_dice_coefficient']):.4f}
    - Best validation IoU: {max(history.history['val_iou_metric']):.4f}
    - Submission file: {submission_file}
    """)
    else:
        print(f"""
    Summary:
    --------
    - Training skipped (used saved weights)
    - Submission file: {submission_file}
    """)

    
    return model, history

if __name__ == "__main__":
    model, history = main()
