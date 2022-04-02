import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python, conv2d_nhwc_python

@auto_scheduler.register_workload
def conv2dbiasrelu_nchw_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_nhwc_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, H, W, CI), name="data", dtype="float16")
    kernel = te.placeholder((KH, KW, CI, CO), name="kernel", dtype="float16")
    conv = topi.nn.conv2d_nhwc(data, kernel, stride, padding, dilation=1, out_dtype="float16")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_nchw_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data", dtype="float16")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel", dtype="float16")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float16")
    return [data, kernel, conv]

def test_conv2dbiasrelu_nchw(sch, args, target, N, H, W, CO, CI, KH, KW, stride, padding):
    func = tvm.build(sch, args, target)

    # Check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
    conv_np = conv2d_nchw_python(data_np, weight_np, stride, padding)
    out_np = np.maximum(conv_np + bias_np, 0.0)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)
    func(data_tvm, weight_tvm, bias_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

def test_conv2d_nhwc(sch, args, target, N, H, W, CO, CI, KH, KW, stride, padding):
    func = tvm.build(sch, args, target)

    # Check correctness
    data_np = np.random.uniform(size=(N, H, W, CI)).astype(np.float16)
    weight_np = np.random.uniform(size=(KH, KW, CI, CO)).astype(np.float16)
    conv_np = conv2d_nhwc_python(data_np, weight_np, stride, padding)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    conv_tvm = tvm.nd.array(np.zeros(conv_np.shape).astype(np.float16), device=dev)
    func(data_tvm, weight_tvm, conv_tvm)

    # Check results
    np.testing.assert_allclose(conv_np, conv_tvm.numpy(), rtol=1e-1)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=600)
    print("\nN: %d, H: %d, W: %d, iC: %d, oC: %d, kW: %d, kH: %d" % (N, H, W, CI, CO, KH, KW))
    print(
        "\nExecution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, conv_tvm).results) * 1000)
    )

def test_conv2d_nchw(sch, args, target, N, H, W, CO, CI, KH, KW, stride, padding):
    func = tvm.build(sch, args, target)

    # Check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float16)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float16)
    conv_np = conv2d_nchw_python(data_np, weight_np, stride, padding)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    conv_tvm = tvm.nd.array(np.zeros(conv_np.shape).astype(np.float16), device=dev)
    func(data_tvm, weight_tvm, conv_tvm)

    # Check results
    np.testing.assert_allclose(conv_np, conv_tvm.numpy(), rtol=1e-1)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=600)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, conv_tvm).results) * 1000)
    )


params = [
    [224, 224, 3, 64, 7, 7, (2, 2), (3, 3)],  # 1
    [56, 56, 64, 64, 3, 3, (1, 1), (1, 1)],   # 2
    [56, 56, 64, 128, 1, 1, (2, 2), (0, 0)],  # 3
    [56, 56, 64, 128, 3, 3, (2, 2), (1, 1)],  # 4
    [28, 28, 128, 128, 3, 3, (1, 1), (1, 1)], # 5
    [14, 14, 256, 256, 3, 3, (1, 1), (1, 1)], # 6
    [14, 14, 256, 512, 1, 1, (2, 2), (0, 0)], # 7
    [14, 14, 256, 512, 3, 3, (2, 2), (1, 1)], # 8
    [7, 7, 512, 512, 3, 3, (1, 1), (1, 1)]    # 9
]

if __name__ == "__main__":
    target = tvm.target.Target("cuda")

    N = 128
    IH, IW, CI, CO, KH, KW, strides, padding = params[1]

    task = auto_scheduler.SearchTask(
        func=conv2d_nhwc_layer, args=(N, IH, IW, CO, CI, KH, KW, strides, padding), target=target
    )
    # Inspect the computational graph
    # print("Computational DAG:")
    # print(task.compute_dag)

    log_file = "log.json"
    os.system("rm " + log_file)
    # Run auto-tuning (search)
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=0)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )
    task.tune(tune_option)
    del measure_ctx
    
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))

    # test
    test_conv2d_nhwc(sch, args, target, N, IH, IW, CO, CI, KH, KW, strides, padding)

    # print("CUDA source code:")
    # print(task.print_best(log_file, print_mode="cuda"))
