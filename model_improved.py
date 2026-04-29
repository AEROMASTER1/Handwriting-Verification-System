# src/model_improved.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model

IMG_SHAPE = (105,105,1)

def build_base_cnn():
    # Use a pretrained MobileNetV2 backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=(105,105,3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base.trainable = False   # freeze base layers
    return base

def build_siamese():
    # Convert grayscale input to RGB (MobileNetV2 requires 3 channels)
    def to_rgb(x):
        return tf.image.grayscale_to_rgb(x)

    input_a = Input(shape=IMG_SHAPE)
    input_b = Input(shape=IMG_SHAPE)

    rgb_a = Lambda(to_rgb)(input_a)
    rgb_b = Lambda(to_rgb)(input_b)

    base = build_base_cnn()

    feat_a = base(rgb_a)
    feat_b = base(rgb_b)

    diff = Lambda(lambda x: tf.abs(x[0] - x[1]))([feat_a, feat_b])

    output = Dense(1, activation='sigmoid')(diff)

    return Model(inputs=[input_a, input_b], outputs=output)
