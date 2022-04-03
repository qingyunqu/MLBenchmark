import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    # C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j], name="out")

    return [A, B, out]

if __name__ == "__main__":
    target = tvm.target.Target("cuda")

    M, N, K = 512, 512, 512
    task = auto_scheduler.SearchTask(
        func=matmul, args=(M, N, K, "float16"), target=target
    )
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "log1.json"
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


    func = tvm.build(sch, args, target)

    lhs_np = np.random.uniform(size=(M, K)).astype(np.float16)
    rhs_np = np.random.uniform(size=(K, N)).astype(np.float16)
    out_np = np.random.uniform(size=(M, N)).astype(np.float16)

    dev = tvm.cuda()
    lhs_tvm = tvm.nd.array(lhs_np, device=dev)
    rhs_tvm = tvm.nd.array(rhs_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)
    func(lhs_tvm, rhs_tvm, out_tvm)

    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=600)
    print("\nM: %d, N: %d, K: %d" % (M, N, K))
    print(
        "\nExecution time of this operator: %.3f ms"
        % (np.median(evaluator(lhs_tvm, rhs_tvm, out_tvm).results) * 1000)
    )
