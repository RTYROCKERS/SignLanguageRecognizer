from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf

# Load the model
model = load_model("Model/keras_model.h5")

# Modify the layers if necessary
for layer in model.layers:
    if isinstance(layer, DepthwiseConv2D):
        # Remove the 'groups' argument and use the default one
        config = layer.get_config()
        if 'groups' in config:
            del config['groups']
        new_layer = DepthwiseConv2D(**config)
        model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=lambda layer: new_layer if layer == old_layer else layer)

# Save the modified model
model.save("Model/modified_model.h5")
