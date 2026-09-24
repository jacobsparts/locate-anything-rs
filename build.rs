//! The engine no longer compiles or embeds the whole toolkit kernel set: it
//! declares the kernels it actually calls and nvcc generates code for those
//! only (`--entries`), so nothing unused reaches the binary.

/// Every kernel this engine resolves by name. Kept in one place so it can be
/// checked against the toolkit before nvcc runs: an unknown name here would
/// otherwise fail at module load instead of at build time.
const KERNELS: &[&str] = &[
    // elementwise / norm / rope
    "lg_noop",
    "lg_rms_norm",
    "lg_layer_norm",
    "lg_rope_neox",
    "lg_rope_2d",
    "lg_silu_mul",
    "lg_gelu_tanh",
    "lg_gelu_erf",
    "lg_add_inplace",
    "lg_row_affine",
    // attention
    "lg_attn_gqa",
    "lg_attn_flash",
    // GEMM / quantized GEMM
    "lg_f32_gemm",
    "lg_f32_gemm_tiled",
    "lg_quantize_q8_0",
    "lg_q8_0_gemm_dp4a",
    "lg_q8_0_gemm_aligned",
    "lg_q8_0_gemv",
    // gather / scatter / decode helpers
    "lg_extract_rows",
    "lg_get_row_q8_0_aligned",
    "lg_copy_row",
    "lg_merge_2x2",
    "lg_set_i32",
    "lg_argmax",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Fail the build, not the run, on a kernel name the toolkit does not define.
    for k in KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit"
        );
    }

    let src = lightgpu_build::toolkit_kernels_cu()
        .expect("locate the toolkit's cuda/kernels.cu (set LA_GPU_DIR to override)");
    lightgpu_build::fatbin_entries(&src.to_string_lossy(), "la_kernels.fatbin", KERNELS);
}
