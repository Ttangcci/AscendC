import numpy as np
import sys
import os

def gen_data(shape, dtype):
    batch = shape[0]
    num_classes = shape[1]

    if dtype == 'bfloat16':
        import torch
        import struct
        features_f32 = np.random.uniform(-1, 1, shape).astype(np.float32)
        labels_f32 = np.zeros(shape, dtype=np.float32)
        for i in range(batch):
            labels_f32[i][np.random.randint(0, num_classes)] = 1.0

        shifted = features_f32 - features_f32.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        softmax = exp_x / exp_x.sum(axis=-1, keepdims=True)
        loss_f32 = (-np.sum(labels_f32 * np.log(softmax + 1e-10), axis=-1)).astype(np.float32)
        backprop_f32 = (softmax - labels_f32).astype(np.float32)

        def f32_to_bf16_bytes(arr):
            f32_bytes = arr.flatten().tobytes()
            bf16_bytes = bytearray()
            for i in range(0, len(f32_bytes), 4):
                bf16_bytes += f32_bytes[i+2:i+4]  # 取高2字节
            return bytes(bf16_bytes)

        open("bfloat16_features.bin", "wb").write(f32_to_bf16_bytes(features_f32))
        open("bfloat16_labels.bin", "wb").write(f32_to_bf16_bytes(labels_f32))
        open("bfloat16_loss_expect.bin", "wb").write(f32_to_bf16_bytes(loss_f32))
        open("bfloat16_backprop_expect.bin", "wb").write(f32_to_bf16_bytes(backprop_f32))
        print(f"Generated data for shape={shape}, dtype={dtype}")
        return

    features = np.random.uniform(-1, 1, shape).astype(dtype)
    labels = np.zeros(shape, dtype=dtype)
    for i in range(batch):
        labels[i][np.random.randint(0, num_classes)] = 1.0

    features_f32 = features.astype(np.float32)
    labels_f32 = labels.astype(np.float32)
    shifted = features_f32 - features_f32.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    softmax = exp_x / exp_x.sum(axis=-1, keepdims=True)
    loss = (-np.sum(labels_f32 * np.log(softmax + 1e-10), axis=-1)).astype(dtype)
    backprop = (softmax - labels_f32).astype(dtype)

    features.tofile(f"{dtype}_features.bin")
    labels.tofile(f"{dtype}_labels.bin")
    loss.tofile(f"{dtype}_loss_expect.bin")
    backprop.tofile(f"{dtype}_backprop_expect.bin")
    print(f"Generated data for shape={shape}, dtype={dtype}")

if __name__ == "__main__":
    shape = eval(sys.argv[1])
    dtype = sys.argv[2]
    gen_data(shape, dtype)