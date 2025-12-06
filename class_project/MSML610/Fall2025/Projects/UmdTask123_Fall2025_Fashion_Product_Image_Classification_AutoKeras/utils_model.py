import tensorflow as tf
import autokeras as ak

def make_baseline_cnn(input_shape=(224, 224, 3), num_classes=6):
    """
    Build a simple CNN baseline used for comparison with AutoKeras.
    """

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0/255)(inputs)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_autokeras_image_classifier(num_classes=6, max_trials=2):
    """
    Create an AutoKeras ImageClassifier searcher.
    This does *not* train; training happens via clf.fit().
    """

    clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=max_trials,
        num_classes=num_classes,
    )
    return clf
