import tensorflow as tf
import os


def pb2tflite(weights):
    fn, ext = os.path.splitext(os.path.normpath(weights).split(os.sep)[-1])

    converter = tf.lite.TFLiteConverter.from_saved_model(weights.replace(ext, ''), signature_keys=['serving_default'])
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_quant_model = converter.convert()

    with open(weights.replace(ext, '.tflite'), 'wb') as f_w:
        f_w.write(tflite_quant_model)
