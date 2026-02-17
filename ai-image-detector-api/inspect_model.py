import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

MODEL_PATH = 'model/ai_image_detector_final.keras'

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.summary()
        print("\nInput shape:", model.input_shape)
        print("Output shape:", model.output_shape)
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Model not found at {MODEL_PATH}")
