//! Compiles the kernels this engine needs into ONE module.
//!
//! Every op here comes from the shared `lightgpu` toolkit, including the
//! batched-GEMM prefill attention (`lg_attn_prefill_*`), which was written
//! engine-local first and promoted once it measured faster - see that kernel's
//! comment in the toolkit for the numbers and for why the obvious version of it
//! is not the fast one.
//!
//! The list is checked against the toolkit's `cuda/kernels.cu` before nvcc runs,
//! so a typo or a name the toolkit renamed away fails the build rather than the
//! first forward pass. The other direction matters too: a kernel the toolkit
//! gains is not compiled here unless it is listed, which is what keeps this
//! engine's fatbin from growing on its own.

/// Generic ops that live in the shared toolkit.
const TOOLKIT_KERNELS: &[&str] = &[
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
    // attention: decode (warp per query token) and the batched-GEMM prefill
    // form (scores -> row softmax -> PV), which the ViT also uses via
    // query tiles
    "lg_attn_gqa",
    "lg_attn_gqa_sk_p1",
    "lg_attn_gqa_sk_p2",
    "lg_attn_prefill_scores",
    "lg_attn_prefill_softmax",
    "lg_attn_prefill_out",
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
    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit"
        );
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate the toolkit's cuda/kernels.cu (set LA_GPU_DIR to override)");

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit.to_string_lossy(),
            out_name: "la_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
    ]);
}
