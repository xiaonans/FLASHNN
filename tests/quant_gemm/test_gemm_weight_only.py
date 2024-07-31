# Copyright 2024 The FLASHNN Authors. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import flashnn
import torch
import triton

torch.manual_seed(0)

def block_quantize(x, scales, block_size, quant_min, quant_max, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.int8)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    k = x.shape[0]
    inters_in_k = scales.shape[0]
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    assert k // block_size == inters_in_k
    for i in range(inters_in_k):
        k_start = i * block_size
        k_end = (i + 1) * block_size
        x_in_iter = x[k_start:k_end, :]
        scale_in_iter = scales[i]
        zero_points_in_iter = zero_points[i]
        x_in_iter_dequant = (
            (x_in_iter / scale_in_iter + zero_points_in_iter)
            .round()
            .clamp(quant_min, quant_max)
        )
        res[k_start:k_end, :] = x_in_iter_dequant
    return res


def channel_quantize(x, scales, quant_min, quant_max, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.int8)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    q_x = (x / scales + zero_points).round().clamp(quant_min, quant_max)
    res.copy_(q_x)
    return res


def block_dequantize(x, scales, block_size, zero_points=None):
    res = torch.zeros_like(x, dtype=torch.half)
    assert x.dim() == 2, "block_fake_quant only support tensor with dim=2."
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    k = x.shape[0]
    inters_in_k = scales.shape[0]
    assert k // block_size == inters_in_k
    for i in range(inters_in_k):
        k_start = i * block_size
        k_end = (i + 1) * block_size
        x_in_iter = x[k_start:k_end, :]
        scale_in_iter = scales[i]
        zero_points_in_iter = zero_points[i]
        x_in_iter_dequant = (x_in_iter - zero_points_in_iter) * scale_in_iter
        res[k_start:k_end, :] = x_in_iter_dequant
    return res


def channel_dequantize(x, scales, zero_points=None):
    if zero_points is None:
        zero_points = torch.zeros_like(scales)
    return (x - zero_points) * scales


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)

def down_size_(size, scale):
    assert size[-1] % scale == 0, f"{size} last dim not divisible by {scale}"
    return (*size[:-1], size[-1] // scale)

def pack_int8_tensor_to_packed_int4(int8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = int8_data.shape
    assert shape[-1] % 2 == 0
    int8_data = int8_data.contiguous().view(-1)
    return (int8_data[::2] << 4 | int8_data[1::2]).view(down_size_(shape, 2))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TestWeightOnlyQGemm(unittest.TestCase):
    def _run_test_weight_layout_transform(
        self,
        gemm_m,
        gemm_n,
        gemm_k,
        compute_type,
        weight_dtype,
        use_bias=False,
        is_sub_channel=False,
        quant_method="symmetric",
    ):
        weights = random_tensor(
            (gemm_k, gemm_n), dtype=compute_type, device="cuda", mean=0, std=0.002
        )
        # weights= torch.randn(gemm_k, gemm_n, dtype=compute_type, device="cuda")

        inputs = torch.randn(gemm_m, gemm_k, dtype=compute_type, device="cuda")
        bias = torch.randn(gemm_n, dtype=compute_type, device="cuda")

        if weight_dtype == torch.int8:
            quant_min, quant_max = -128, 127
        elif weight_dtype == torch.quint4x2:
            quant_min, quant_max = -8, 7
        else:
            raise RuntimeError("Unsupported weight dtype")
        zero_points = None
        k_per_scale = 0
        if not is_sub_channel:
            qscheme = torch.per_channel_symmetric
            if quant_method == "asymmetric":
                qscheme = torch.per_channel_affine
            min_max_observer = PerChannelMinMaxObserver.with_args(
                ch_axis=1,
                quant_min=quant_min,
                quant_max=quant_max,
                qscheme=qscheme,
            )

            obs = min_max_observer().to(compute_type).cuda()
            obs(weights)
            if quant_method == "asymmetric":
                scales, zero_points = obs.calculate_qparams()
                zero_points = zero_points.to(scales.dtype)
            else:
                scales, _ = obs.calculate_qparams()
                zero_points = None
            q_weights = channel_quantize(
                weights, scales.unsqueeze(0), quant_min, quant_max, zero_points
            )
            if quant_method == "asymmetric":
                unsqueeze_zero_points = zero_points.unsqueeze(0)
            else:
                unsqueeze_zero_points = None
            dq_weights = channel_dequantize(
                q_weights, scales.unsqueeze(0), zero_points=unsqueeze_zero_points
            )
        else:
            k_per_scale = 64
            scale_k = gemm_k // k_per_scale
            scales = torch.randn(scale_k, gemm_n, dtype=compute_type, device="cuda")
            zero_points = None
            if quant_method == "asymmetric":
                zero_points = torch.randint_like(scales, low=quant_min, high=quant_max)
            q_weights = block_quantize(
                weights, scales, k_per_scale, quant_min, quant_max, zero_points
            )
            dq_weights = block_dequantize(q_weights, scales, k_per_scale, zero_points)
        reference_result = torch.matmul(inputs, dq_weights)
        q_weights = q_weights.cpu()
        if weight_dtype == torch.quint4x2:
            q_weights = pack_int8_tensor_to_packed_int4(q_weights)
        q_weights = q_weights.permute(1, 0).contiguous().cuda()
        if use_bias:
            reference_result += bias.unsqueeze(0)
        else:
            bias = None
        
        gemm_weight_only = flashnn.GemmWeightOnly()
        tri_result = gemm_weight_only(
            inputs, q_weights, scales, bias, zero_points
        )
        print("found best config: ", gemm_weight_only.best_config)

        ms_torch = triton.testing.do_bench(lambda: torch.matmul(inputs, dq_weights))
        ms_triton = triton.testing.do_bench(lambda: gemm_weight_only(
            inputs, q_weights, scales, bias, zero_points))
        perf = lambda ms: 2 * gemm_m * gemm_n * gemm_k * 1e-12 / (ms * 1e-3)

        print(bcolors.WARNING + "torch:", "%0.2f"%perf(ms_torch), "TFLOPS" + bcolors.ENDC)
        print(bcolors.WARNING + "triton:", "%0.2f"%perf(ms_triton), "TFLOPS" + bcolors.ENDC)
        # torch.testing.assert_close(
        #     tri_result, reference_result, rtol=0.001, atol=0.002, check_dtype=False
        # )

    def test_weight_layout_transform(self):
        gemm_m, gemm_n, gemm_k = 4096, 4096, 4096
        gemm_m = [1, 32,512, 4096]
        gemm_k_n = [(4096, 4096), (4096, 1024), (4096, 14336), (14336, 4096)]
        compute_type = torch.float16
        weight_dtype = [torch.int8] #torch.quint4x2]
        use_bias = [False]
        is_sub_channel = [True]
        quant_methods = ["asymmetric"]
        for quant_method in quant_methods:
            for w in weight_dtype:
                for b in use_bias:
                    for s in is_sub_channel:
                        for m in gemm_m:
                            for k, n in gemm_k_n:
                                print(
                                    f"\nweight_dtype={w} use_bias={b} is_sub_channel={s} quant_method={quant_method}, M={m}, N={n}, K={k}.....",
                                    flush=True,
                                )
                                self._run_test_weight_layout_transform(
                                    m, n, k, compute_type, w, b, s, quant_method
                                )
                                print(
                                    f"Pass test of weight_layout_transform with weight_dtype={w} use_bias={b} is_sub_channel={s} quant_method={quant_method}!",
                                    flush=True,
                                )


if __name__ == "__main__":
    unittest.main()
