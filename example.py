import subprocess

import numpy as np
import onnxruntime as ort
import torch


class SomethingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = torch.tensor([-1, 2], dtype=torch.float32)  # Constant node
        z = torch.randn(2, dtype=torch.float32)  # RandomNormal node
        return x + y + z


def export_onnx(path: str) -> None:
    x = torch.tensor([10, 20], dtype=torch.float32)
    model = SomethingModel()
    model.eval()

    torch.onnx.export(
        model, x, path,
        verbose=True,
        export_params=True,
        do_constant_folding=False,
        opset_version=11,
        input_names=['x'],
        output_names=['output'],
    )


def main():
    export_onnx("before.onnx")

    retcode = subprocess.call(["python", "onnx_remove_random_normal.py", "before.onnx", "after.onnx"])
    assert retcode == 0

    ort_session = ort.InferenceSession("after.onnx")
    model_inputs = ort_session.get_inputs()[0]
    assert model_inputs.name == "x"
    shape = model_inputs.shape

    x = np.random.randn(*shape).astype('float32')
    print("x", x)

    outputs = ort_session.run(None, {"x": x})
    print("outputs", outputs)


if __name__ == '__main__':
    main()
