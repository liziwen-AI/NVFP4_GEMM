import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from task import input_t, output_t
from utils import make_match_reference
from contestant.submission import custom_kernel
# from contestant.gau_nernst import custom_kernel
# from contestant.CatsRCool import custom_kernel
# from contestant.s_am import custom_kernel

# Scaling factor vector size
sf_vec_size = 16

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMM result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
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


def generate_input(
    m: int,
    n: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMM.
    
    Args:
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Number of columns in A and rows of B
        l: Batch size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [n, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, n, l] - Output matrix in torch.float16 data type
    """
    torch.manual_seed(seed)
    
    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        -128, 128, (l, m, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b_ref = torch.randint(
        -128, 128, (l, n, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )
    
    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(0, 4, ref_shape, dtype=torch.int8, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(0, 4, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')
        
        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
        
        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k
        
        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        
        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    return (a_ref, b_ref, sfa_ref_cpu.to("cuda"), sfb_ref_cpu.to("cuda"), sfa_ref_permuted, sfb_ref_permuted, c_ref)

def benchmark(data, warmup=10, iters=100, l2_flush_size_mb=512):
    # nvidia-smi -lgc 1500
    numel = (l2_flush_size_mb * 1024 * 1024) // 4
    l2_flush_buffer = torch.empty(numel, dtype=torch.float32, device="cuda")

    def _flush_l2():
        l2_flush_buffer.fill_(1.0)

    my_output = custom_kernel(data)
    is_correct, error_msg = check_implementation(data, my_output)
    torch.cuda.synchronize()
    if is_correct:
        print("🎉 恭喜！你的自定义算子实现与参考实现完全吻合，精度达标！")
    else:
        print(f"❌ 验证失败！错误信息：{error_msg}")

    _flush_l2()
    torch.cuda.synchronize()
    nvtx.range_push("gemm_profile_range")
    out = custom_kernel(data)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # warmup
    for _ in range(warmup):
        _flush_l2()
        torch.cuda.synchronize() 
        out = custom_kernel(data)
    torch.cuda.synchronize()
    
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        _flush_l2()
        torch.cuda.synchronize()
        starts[i].record()
        out = custom_kernel(data)
        ends[i].record()

    torch.cuda.synchronize()
    

    # 5. 更全面的统计维度
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    
    mean_ms = np.mean(times_ms)
    median_ms = np.median(times_ms)
    min_ms = np.min(times_ms)
    std_ms = np.std(times_ms)
    
    print(f"📊 Benchmark 结果:")
    print(f"   - Average (Mean): {mean_ms:.4f} ms")
    print(f"   - Median:         {median_ms:.4f} ms")
    print(f"   - Minimum:        {min_ms:.4f} ms")
    print(f"   - Std Dev:        {std_ms:.4f} ms")
    
    return median_ms

# 构造测试数据（需要看 task.py 里 input_t 的定义）
# 通常 data = (A, B, ..., SFA, SFB, C)

def benchmark_new(data, warmup=10, iters=100, flush_l2=True, l2_flush_size_mb=256):
    """
    对 custom_kernel 做正确性验证 + 性能测试。

    改进点：
    1. 正确性检查只做一次，避免重复。
    2. 每次迭代前可选地 flush L2 cache，避免多次迭代命中同一份数据的 cache，
       导致测出的耗时比真实场景（cache miss）偏乐观。
    3. 用每次迭代单独的 CUDA Event 记录耗时，最后统计中位数/均值/标准差，
       而不是只给一个粗略的总时间/iters。
    4. 顺带算出 TFLOPS，方便和 roofline 模型对比。
    """
    # ---- 正确性验证（只做一次） ----
    my_output = custom_kernel(data)
    is_correct, error_msg = check_implementation(data, my_output)
    if is_correct:
        print("🎉 恭喜！你的自定义算子实现与参考实现完全吻合，精度达标！")
    else:
        print(f"❌ 验证失败！错误信息：{error_msg}")
        return None

    # ---- 用于 flush L2 cache 的 dummy buffer ----
    # 思路：在每次 kernel 调用前，读写一块明显大于 L2 cache 容量的显存，
    # 把 A/B/C 等真正关心的数据从 L2 中挤出去，模拟真实场景下的 cache miss。
    l2_flush_buffer = None
    if flush_l2:
        numel = (l2_flush_size_mb * 1024 * 1024) // 4  # float32 元素个数
        l2_flush_buffer = torch.empty(numel, dtype=torch.float32, device="cuda")

    def _flush_l2():
        if l2_flush_buffer is not None:
            l2_flush_buffer.fill_(1.0)

    # ---- warmup ----
    for _ in range(warmup):
        _flush_l2()
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    # ---- 正式计时：每次迭代单独记录 ----
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        _flush_l2()
        start_events[i].record()
        _ = custom_kernel(data)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times_ms.sort()

    mean_ms = sum(times_ms) / len(times_ms)
    median_ms = times_ms[len(times_ms) // 2]
    p90_ms = times_ms[int(len(times_ms) * 0.9)]
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    # ---- 计算 TFLOPS（GEMM: 2*M*N*K*L 次浮点运算） ----
    a_ref, b_ref, _, _, _, _, c_ref = data
    m_dim, k_dim, l_dim = a_ref.shape
    n_dim = b_ref.shape[0]
    flops = 2 * m_dim * n_dim * k_dim * l_dim
    tflops_mean = flops / (mean_ms * 1e-3) / 1e12
    tflops_median = flops / (median_ms * 1e-3) / 1e12

    print(f"迭代次数: {iters}  (L2 flush: {'开启' if flush_l2 else '关闭'})")
    print(f"耗时(ms): mean={mean_ms:.4f}  median={median_ms:.4f}  "
          f"min={min_ms:.4f}  max={max_ms:.4f}  p90={p90_ms:.4f}  std={std_ms:.4f}")
    print(f"TFLOPS: mean={tflops_mean:.2f}  median={tflops_median:.2f}")

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p90_ms": p90_ms,
        "std_ms": std_ms,
        "tflops_mean": tflops_mean,
        "tflops_median": tflops_median,
    }

check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)

m, n, k, l, seed = 2304, 4608, 7168, 1, 1111
data = generate_input(m, n, k, l, seed)


# my_output = custom_kernel(data)
# is_correct, error_msg = check_implementation(data, my_output)
# if is_correct:
#     print("🎉 恭喜！你的自定义算子实现与参考实现完全吻合，精度达标！")
# else:
#     print(f"❌ 验证失败！错误信息：{error_msg}")

ms = benchmark(data)
print(": 💯 ✅ ❌ ⚡ 🎯 🚀")  
print(f"custom_kernel time: {ms}")  


