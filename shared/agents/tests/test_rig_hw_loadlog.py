import sys
import unittest
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import hw_loadlog  # noqa: E402

# Ornith-1.5-9B on GPUs 1+3, llama.cpp b11669, 2026-10-18 (trimmed startup log).
FULL = """\
0.00.718.896 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce GTX 1060 6GB):   6065 total,   5501 used,    465 free vs. target of    512
0.00.718.897 I common_params_fit_impl:   - CUDA1 (NVIDIA GeForce GTX 1060 6GB):   6065 total,   6053 used,    -81 free vs. target of    512
0.00.718.898 I common_params_fit_impl: cannot meet free memory targets on all devices, need to use 640 MiB less in total
0.01.163.364 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to -2, abort
1.05.402.061 I load_tensors: offloaded 34/34 layers to GPU
1.05.402.072 I load_tensors:   CPU_Mapped model buffer size =   545.62 MiB
1.05.402.076 I load_tensors:        CUDA0 model buffer size =  2125.56 MiB
1.05.402.079 I load_tensors:        CUDA1 model buffer size =  2686.69 MiB
0.27.889.235 I llama_context: n_ctx                 = 248064
0.27.893.702 I llama_kv_cache:      CUDA0 KV buffer size =  2059.12 MiB
0.27.907.085 I llama_kv_cache:      CUDA1 KV buffer size =  2059.12 MiB
0.27.979.367 I sched_reserve:      CUDA0 compute buffer size =  1111.20 MiB
0.27.979.373 I sched_reserve:      CUDA1 compute buffer size =  1102.20 MiB
0.27.979.374 I sched_reserve:  CUDA_Host compute buffer size =   132.20 MiB
"""
PARTIAL = "0.04.608.367 I load_tensors: offloaded 33/34 layers to GPU\n"


class LoadLogTests(unittest.TestCase):
    def test_reads_offload_ctx_buffers_and_projection_with_gpu_ids(self):
        info = hw_loadlog.parse(FULL, [1, 3])
        self.assertTrue(info["full_offload"])
        self.assertEqual((info["offloaded"], info["layers"], info["n_ctx"]), (34, 34, 248064))
        self.assertEqual([c["gpu"] for c in info["cards"]], [1, 3])
        self.assertEqual(info["cards"][1], {"gpu": 3, "model_mb": 2687, "kv_mb": 2059, "compute_mb": 1102,
                                            "total_mb": 5848})
        self.assertEqual(info["host_mb"], 678)
        self.assertEqual([p["gpu"] for p in info["projected_short"]], [3])
        self.assertIn("GPU 3 6053 of 6065 MiB (81 MiB over)", hw_loadlog.summary(info))

    def test_partial_offload_is_reported(self):
        info = hw_loadlog.parse(PARTIAL, [1, 3])
        self.assertFalse(info["full_offload"])
        self.assertIn("only 33/34 layers", hw_loadlog.summary(info))

    def test_empty_log(self):
        info = hw_loadlog.parse("", [0])
        self.assertIsNone(info["offloaded"])
        self.assertFalse(info["full_offload"])
        self.assertEqual(hw_loadlog.summary(info), "no fit information in the log")


if __name__ == "__main__":
    unittest.main()
