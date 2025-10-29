import tensorflow
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.applications
import tensorflow.keras.optimizers
import wandb
# import wandb.keras
import math
import os
import dotenv
import dataset_operations # Import the script with main_prep()

# --- W&B Setup ---
dotenv.load_dotenv()
# Note: Using os.environ.get to fetch the key
key = os.environ.get("WANDB_API_KEY") 

if key:
    print("Logging into Weights & Biases...")
    wandb.login(key=key)
else:
    print("WARNING: WANDB_API_KEY not found in .env. Logging in anonymously.")
    wandb.login()

# Initialize the project run
wandb.init(project="animal-faces-classification") 
# -----------------

# --- Configuration for Training ---
NUM_EPOCHS = 10 
LEARNING_RATE = 0.0001


def build_and_train_model(train_generator, val_generator, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """
    Builds a ResNet50 Transfer Learning model, compiles it, and trains it 
    while logging metrics to the already-initialized Weights & Biases run.
    """
    print("\n--- Initializing Transfer Learning Pipeline (ResNet50) ---")
    
    # --- 1. Model Selection and Setup ---
    
    # Get necessary parameters from the generators
    IMG_HEIGHT, IMG_WIDTH = train_generator.target_size
    num_classes = len(train_generator.class_indices)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    print(f"Model Input Shape: {input_shape}")
    print(f"Number of Classes: {num_classes}")
    
    # Load ResNet50 base model (Transfer Learning)
    base_model = tensorflow.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create the top layers for classification
    x = base_model.output
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tensorflow.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    print("ResNet50 Base Model loaded and frozen. Top layers added.")

    # Log configuration to the already-initialized W&B run
    wandb.config.update({
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": train_generator.batch_size,
        "model_type": "ResNet50_Transfer",
        "dataset": "AFHQ"
    })
    
    # --- 2. Compilation ---
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate steps per epoch
    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    val_steps = math.ceil(val_generator.samples / val_generator.batch_size)
    
    # --- 3. Model Training and Tracking ---

    print("\nStarting model training...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[
            # The W&B Keras Callback tracks all metrics, model topology, and system stats
            wandb.keras.WandbCallback() 
        ]
    )

    # --- 4. Final Evaluation ---
    print("\n--- Final Model Evaluation ---")
    final_loss, final_accuracy = model.evaluate(val_generator, steps=val_steps)
    print(f"Validation Loss: {final_loss:.4f}")
    print(f"Validation Accuracy: {final_accuracy:.4f}")
    
    wandb.log({
        "final_val_accuracy": final_accuracy,
        "final_val_loss": final_loss
    })

    # The wandb.finish() is called in the __main__ block for cleaner exit
    return model


# --- Main Execution ---
if __name__ == "__main__":
    
    try:
        print("--- Calling Data Preparation Pipeline from dataset_operations.py ---")
        # Call the data preparation function to get the generators
        train_gen, val_gen = dataset_operations.main_prep()

        print("\n--- Starting Model Training ---")
        # Start the training process
        final_model = build_and_train_model(train_gen, val_gen)
        
    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
    finally:
        # Ensure the W&B run is closed even if an error occurs during training
        if wandb.run:
            wandb.finish()