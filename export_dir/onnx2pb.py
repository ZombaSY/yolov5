import onnx
import os
from onnx_tf.backend import prepare


def onnx2pb(weights):
    fn, ext = os.path.splitext(os.path.normpath(weights).split(os.sep)[-1])
    weights_new = weights.replace(ext, '')

    onnx_model = onnx.load(weights.replace(ext, '.onnx'))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(weights_new)
