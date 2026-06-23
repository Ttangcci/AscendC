import numpy as np
import sys

def compare(dtype):
    if dtype == 'float32':
        np_dtype = np.float32
        rtol, atol = 1e-4, 1e-4
    elif dtype == 'bfloat16':
        def bf16_bytes_to_f32(filename):
            data = open(filename, "rb").read()
            f32_bytes = bytearray()
            for i in range(0, len(data), 2):
                f32_bytes += b'\x00\x00' + data[i:i+2]
            return np.frombuffer(bytes(f32_bytes), dtype=np.float32)

        loss_expect  = bf16_bytes_to_f32("bfloat16_loss_expect.bin")
        loss_actual  = bf16_bytes_to_f32("bfloat16_loss_actual.bin")
        bp_expect    = bf16_bytes_to_f32("bfloat16_backprop_expect.bin")
        bp_actual    = bf16_bytes_to_f32("bfloat16_backprop_actual.bin")

        loss_ok = np.allclose(loss_actual, loss_expect, rtol=1e-1, atol=1e-1)
        bp_ok   = np.allclose(bp_actual,   bp_expect,   rtol=1e-1, atol=1e-1)
        if loss_ok and bp_ok:
            print(f"[PASS] bfloat16 loss and backprop match!")
        else:
            if not loss_ok:
                print(f"[FAIL] bfloat16 loss mismatch!")
                print(f"  expect: {loss_expect[:5]}")
                print(f"  actual: {loss_actual[:5]}")
            if not bp_ok:
                print(f"[FAIL] bfloat16 backprop mismatch!")
                print(f"  expect: {bp_expect[:5]}")
                print(f"  actual: {bp_actual[:5]}")
            exit(1)
        return
    else:
        np_dtype = np.float16
        rtol, atol = 1e-2, 1e-2

    loss_expect  = np.fromfile(f"{dtype}_loss_expect.bin",    dtype=np_dtype)
    loss_actual  = np.fromfile(f"{dtype}_loss_actual.bin",    dtype=np_dtype)
    bp_expect    = np.fromfile(f"{dtype}_backprop_expect.bin", dtype=np_dtype)
    bp_actual    = np.fromfile(f"{dtype}_backprop_actual.bin", dtype=np_dtype)

    loss_ok = np.allclose(loss_actual, loss_expect, rtol=rtol, atol=atol)
    bp_ok   = np.allclose(bp_actual,   bp_expect,   rtol=rtol, atol=atol)

    if loss_ok and bp_ok:
        print(f"[PASS] {dtype} loss and backprop match!")
    else:
        if not loss_ok:
            print(f"[FAIL] {dtype} loss mismatch!")
            print(f"  expect: {loss_expect[:5]}")
            print(f"  actual: {loss_actual[:5]}")
        if not bp_ok:
            print(f"[FAIL] {dtype} backprop mismatch!")
            print(f"  expect: {bp_expect[:5]}")
            print(f"  actual: {bp_actual[:5]}")
        exit(1)

if __name__ == "__main__":
    dtype = sys.argv[1]
    compare(dtype)