"""
run_all.py

在 liz.py 的基础上做两个扩展：
1. 尺寸 (m, n, k, l, seed) 不再写死一组，而是从 task.yml 里的
   `tests` + `benchmarks` 两个列表里读出来，全部跑一遍。
2. custom_kernel 不再写死 import 一个，而是把
   submission / gau_nernst / CatsRCool / s_am 四个实现都跑一遍，
   分别做正确性校验 + (可选) 性能测试，最后打印一个汇总表。

用法示例（在有 task.yml / task.py / utils.py 的目录下执行）：

    # 只跑正确性，所有 kernel，所有 tests 尺寸（不含 benchmarks，跑得快）
    python run_all.py --shapes tests

    # 只跑 submission 和 gau_nernst，包含 benchmark 尺寸，且做计时
    python run_all.py --kernels submission gau_nernst --shapes all --bench

    # 只跑 benchmarks 里的 3 组尺寸，四个 kernel 都跑，且计时
    python run_all.py --shapes benchmarks --bench

注意：
- CatsRCool.py 用 load_inline 现场编译 C++/CUDA 扩展，第一次 import 会比较慢
  （几十秒到几分钟），属于正常现象，不是卡死。
- 四个实现里如果某个模块 import 失败（比如环境里没装对应依赖、CUTLASS 路径不对），
  脚本会打印警告并跳过它，不会让整体测试中断。
- 各个 kernel 内部都写回同一个 c 张量，因为 alpha=1, beta=0，c 的初始值不影响
  结果，所以可以放心地在同一份 data 上依次跑不同 kernel。
"""

import argparse
import importlib
import statistics
import sys
from pathlib import Path

import torch
import yaml

from task import input_t, output_t
from utils import make_match_reference

# ------------------------------------------------------------------
# 以下这几段和 liz.py 完全一致：scale factor 处理 / 参考实现 / 造数据
# ------------------------------------------------------------------

sf_vec_size = 16


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def ref_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    _, _, l = c_ref.shape

    for l_idx in range(l):
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, :, l_idx] = res
    return c_ref


def generate_input(m: int, n: int, k: int, l: int, seed: int):
    torch.manual_seed(seed)

    a_ref = torch.randint(
        -128, 128, (l, m, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b_ref = torch.randint(
        -128, 128, (l, n, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    def create_scale_factor_tensors(l, mn, sf_k):
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        ref_f8_random_int = torch.randint(0, 4, ref_shape, dtype=torch.int8, device="cuda")
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        mma_permute_order = (3, 4, 1, 5, 2, 0)
        rand_int_tensor = torch.randint(0, 4, mma_shape, dtype=torch.int8, device="cuda")
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        i_idx = torch.arange(mn, device="cuda")
        j_idx = torch.arange(sf_k, device="cuda")
        b_idx = torch.arange(l, device="cuda")
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[
            i_grid, j_grid, b_grid
        ]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    return (
        a_ref,
        b_ref,
        sfa_ref_cpu.to("cuda"),
        sfb_ref_cpu.to("cuda"),
        sfa_ref_permuted,
        sfb_ref_permuted,
        c_ref,
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)


# ------------------------------------------------------------------
# 新增部分：从 task.yml 读尺寸 + 动态加载多个 kernel 实现
# ------------------------------------------------------------------

# module_name -> 文件里对应的 custom_kernel 入口
KERNEL_MODULES = ["submission", "gau_nernst", "CatsRCool", "s_am"]
KERNEL_MODULES = [f"contestant.{m}" for m in KERNEL_MODULES]


def load_shapes(yaml_path: str = "task.yml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tests", []), cfg.get("benchmarks", [])


def load_kernel(module_name: str):
    """动态 import 一个 kernel 模块，返回它的 custom_kernel 函数。"""
    module = importlib.import_module(module_name)
    fn = getattr(module, "custom_kernel", None)
    if fn is None:
        raise AttributeError(f"{module_name} 里没有 custom_kernel 函数")
    return fn


def benchmark_kernel(custom_kernel, data, warmup=10, iters=100,
                      flush_l2=True, l2_flush_size_mb=512):
    """跟 liz.py 里的 benchmark_new 逻辑一致，只是 custom_kernel 作为参数传入。"""
    l2_flush_buffer = None
    if flush_l2:
        numel = (l2_flush_size_mb * 1024 * 1024) // 4
        l2_flush_buffer = torch.empty(numel, dtype=torch.float32, device="cuda")

    def _flush_l2():
        if l2_flush_buffer is not None:
            l2_flush_buffer.fill_(1.0)

    for _ in range(warmup):
        _flush_l2()
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        _flush_l2()
        start_events[i].record()
        _ = custom_kernel(data)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(start_events[i].elapsed_time(end_events[i]) for i in range(iters))
    mean_ms = statistics.mean(times_ms)
    median_ms = times_ms[len(times_ms) // 2]

    a_ref, b_ref, _, _, _, _, c_ref = data
    m_dim, k_dim, l_dim = a_ref.shape
    n_dim = b_ref.shape[0]
    flops = 2 * m_dim * n_dim * k_dim * l_dim
    tflops_mean = flops / (mean_ms * 1e-3) / 1e12
    tflops_median = flops / (median_ms * 1e-3) / 1e12

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": times_ms[0],
        "max_ms": times_ms[-1],
        "tflops_mean": tflops_mean,
        "tflops_median": tflops_median,
    }


def main():
    parser = argparse.ArgumentParser(description="跑全部 shape x 全部 kernel 实现")
    parser.add_argument(
        "--kernels", nargs="+", default=KERNEL_MODULES,
        choices=KERNEL_MODULES,
        help="要测试的 kernel 模块名（默认全部四个）",
    )
    parser.add_argument(
        "--shapes", choices=["tests", "benchmarks", "all"], default="all",
        help="用 task.yml 里的 tests / benchmarks / 两者都用",
    )
    parser.add_argument(
        "--bench", action="store_true", default=True
        help="除了正确性校验，还做计时（会慢很多）",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--yaml", default="task.yml")
    args = parser.parse_args()

    tests, benchmarks = load_shapes(args.yaml)
    if args.shapes == "tests":
        shapes = tests
    elif args.shapes == "benchmarks":
        shapes = benchmarks
    else:
        shapes = tests + benchmarks

    # 结果汇总：results[module_name][shape_idx] = {"correct":..., "msg":..., "bench":...}
    results = {name: [] for name in args.kernels}

    for module_name in args.kernels:
        print(f"\n{'=' * 70}\n加载 kernel 模块: {module_name.removeprefix('contestant.')}\n{'=' * 70}")
        try:
            custom_kernel = load_kernel(module_name)
        except Exception as e:
            print(f"⚠️  加载 {module_name.removeprefix('contestant.')} 失败，跳过：{e}")
            continue

        for shape in shapes:
            m, n, k, l, seed = shape["m"], shape["n"], shape["k"], shape["l"], shape["seed"]
            tag = f"m={m} n={n} k={k} l={l}"
            data = generate_input(m, n, k, l, seed)

            try:
                out = custom_kernel(data)
                ok, msg = check_implementation(data, out)
            except Exception as e:
                ok, msg = False, f"运行异常: {e}"

            entry = {"shape": shape, "correct": ok, "msg": msg}

            if ok:
                print(f"✅ [{module_name.removeprefix('contestant.')}] {tag}")
            else:
                print(f"❌ [{module_name.removeprefix('contestant.')}] {tag}  错误信息: {msg}")

            if ok and args.bench:
                try:
                    bench = benchmark_kernel(
                        custom_kernel, data, warmup=args.warmup, iters=args.iters
                    )
                    entry["bench"] = bench
                    print(
                        f"    耗时(ms): mean={bench['mean_ms']:.4f} "
                        f"median={bench['median_ms']:.4f}  "
                        f"TFLOPS: mean={bench['tflops_mean']:.2f} "
                        f"median={bench['tflops_median']:.2f}"
                    )
                except Exception as e:
                    print(f"    ⚠️ benchmark 失败: {e}")

            results[module_name].append(entry)

    # -------- 汇总表 --------
    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    header = f"{'shape':<30}" + "".join(f"{k.removeprefix('contestant.'):>14}" for k in results.keys())
    print(header)
    for idx, shape in enumerate(shapes):
        tag = f"m={shape['m']},n={shape['n']},k={shape['k']},l={shape['l']}"
        row = f"{tag:<30}"
        for module_name in results.keys():
            if idx < len(results[module_name]):
                entry = results[module_name][idx]
                if not entry["correct"]:
                    cell = "FAIL"
                elif "bench" in entry:
                    cell = f"{entry['bench']['median_ms']:.3f}ms"
                else:
                    cell = "OK"
            else:
                cell = "N/A"
            row += f"{cell:>14}"
        print(row)


if __name__ == "__main__":
    main()