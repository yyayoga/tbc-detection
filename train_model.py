import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "dataset_chest/train"
VAL_DIR = "dataset_chest/test"
MODEL_SAVE_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "tbc_model.h5")

def main():
    # 1. Create model save directory if not exists
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created directory: {MODEL_SAVE_DIR}")

    # 2. Setup Data Generators with Image Augmentation
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Error: Dataset directories not found at {TRAIN_DIR} or {VAL_DIR}.")
        print(f"Please make sure you have the dataset inside '{TRAIN_DIR}' and '{VAL_DIR}'.")
        return

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary' # binary for Normal vs TBC
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 3. Build Transfer Learning Model using MobileNetV2
    print("Building model...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # 4. Train the Model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # 5. Save the Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")

    # 6. Plot Accuracy and Loss
    print("Plotting results...")
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_result.png')
    print("Training plot saved as training_result.png")
    plt.show()

if __name__ == "__main__":
    main()
