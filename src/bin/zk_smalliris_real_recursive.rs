//! Multi-stage (3 proof) recursive Plonky2 for SmallIris: L1 → verify+conv2 →
//! verify+conv3+GAP.
//!
//! ```bash
//! cargo +nightly run --release --bin zk-smalliris-real-recursive
//! cargo +nightly run --release --bin zk-smalliris-real-recursive -- --layer1-only
//! # If L1 runs out of memory, diagnose first, then tighten value bits and try
//! # single-threaded execution and/or lighter FRI parameters.
//! cargo +nightly run --release --bin zk-smalliris-real-recursive -- --diagnose-bits
//! # Example local/demo configuration with many threads and weaker FRI.
//! RAYON_NUM_THREADS=20 cargo +nightly run --release --bin zk-smalliris-real-recursive -- --demo-fri
//! # If memory is tight, try single-threaded proving as well:
//! # RAYON_NUM_THREADS=1 ... --l1-low-memory-fri
//! ```

use anyhow::Context;
use cnn_zkp::{
    smalliris_real_zk::{
        diagnose_min_value_bits, load_rgb_image_input, synthetic_input_i32, RealSmallIrisExport,
    },
    smalliris_real_zk_recursive::{
        prove_three_stage, public_sums_match_final, smoke_layer1_only, verify_final,
        StagedProveConfig,
    },
};
use plonky2::{
    plonk::circuit_data::VerifierCircuitData, util::serialization::DefaultGateSerializer,
};
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    #[structopt(
        long,
        parse(from_os_str),
        default_value = "fixtures/smalliris_real_i32.json"
    )]
    weights:           PathBuf,
    #[structopt(long, parse(from_os_str))]
    input_image:       Option<PathBuf>,
    /// ReLU/MaxPool range-check width; run `--diagnose-bits` on your input.
    /// Default 36 ≥ typical need (~34).
    #[structopt(long, default_value = "36")]
    value_bits:        usize,
    /// Only build + prove + verify **stage 1** (still heavy). Use if full run
    /// disappears with no error (often OOM).
    #[structopt(long)]
    layer1_only:       bool,
    /// Print suggested `--value-bits` from int forward on the test input, then
    /// exit.
    #[structopt(long)]
    diagnose_bits:     bool,
    /// L1 only: slightly lighter FRI (nominal ~80-bit vs 100-bit). Must still
    /// satisfy Plonky2; tiny prover savings.
    #[structopt(long)]
    l1_low_memory_fri: bool,
    /// **L1+R2+R3**: weaker FRI (~64-bit nominal) for faster proving. Course /
    /// demo only — not production security.
    #[structopt(long)]
    demo_fri:          bool,
}

fn diag(msg: &str) {
    eprintln!("{msg}");
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

fn default_manifest_output_path(weights: &Path, input_image: Option<&Path>) -> PathBuf {
    let parent = weights.parent().unwrap_or_else(|| Path::new("."));
    let stem = weights
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("smalliris_proof_manifest");
    let input_suffix = input_image
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .map(|s| format!("_{s}"))
        .unwrap_or_default();
    parent.join(format!("{stem}{input_suffix}_proof_manifest.json"))
}

fn default_proof_output_path(weights: &Path, input_image: Option<&Path>) -> PathBuf {
    let parent = weights.parent().unwrap_or_else(|| Path::new("."));
    let stem = weights
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("smalliris_recursive_proof");
    let input_suffix = input_image
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .map(|s| format!("_{s}"))
        .unwrap_or_default();
    parent.join(format!("{stem}{input_suffix}_recursive_proof.bin"))
}

fn default_verifier_data_output_path(weights: &Path, input_image: Option<&Path>) -> PathBuf {
    let parent = weights.parent().unwrap_or_else(|| Path::new("."));
    let stem = weights
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("smalliris_recursive_verifier_data");
    let input_suffix = input_image
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .map(|s| format!("_{s}"))
        .unwrap_or_default();
    parent.join(format!("{stem}{input_suffix}_recursive_verifier_data.bin"))
}

fn main() -> anyhow::Result<()> {
    diag(&format!(
        "zk-smalliris-real-recursive start | pid={} | cwd={}",
        std::process::id(),
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "?".into())
    ));

    let opt = Opt::from_args();
    let exp = RealSmallIrisExport::load_path(&opt.weights)?;
    let exp_meta = exp.clone();
    let input = if let Some(path) = &opt.input_image {
        load_rgb_image_input(path, exp.h, exp.w, exp.activation_q_i32())?
    } else {
        synthetic_input_i32(&exp)
    };

    if opt.diagnose_bits {
        let b = diagnose_min_value_bits(&exp, &input);
        diag(&format!(
            "diagnose_min_value_bits = {b} -> suggested next step: --value-bits {b} \
             (if OOM persists, circuit size is likely the main issue rather than \
             bit width)"
        ));
        return Ok(());
    }

    let staged = StagedProveConfig {
        l1_low_memory_fri: opt.l1_low_memory_fri,
        demo_fri:          opt.demo_fri,
    };

    if opt.layer1_only {
        smoke_layer1_only(&exp, opt.value_bits, &input, staged)?;
        return Ok(());
    }

    let (proof, tail_data, sums) = prove_three_stage(exp, opt.value_bits, &input, staged)?;
    assert!(public_sums_match_final(&proof, &sums));
    verify_final(&tail_data, &proof)?;
    let manifest_output_path = default_manifest_output_path(&opt.weights, opt.input_image.as_deref());
    let proof_output_path = default_proof_output_path(&opt.weights, opt.input_image.as_deref());
    let verifier_data_output_path =
        default_verifier_data_output_path(&opt.weights, opt.input_image.as_deref());
    std::fs::write(&proof_output_path, proof.to_bytes())
        .with_context(|| format!("write {}", proof_output_path.display()))?;
    let verifier_data = VerifierCircuitData {
        verifier_only: tail_data.verifier_only.clone(),
        common:        tail_data.common.clone(),
    };
    let verifier_data_bytes = verifier_data
        .to_bytes(&DefaultGateSerializer)
        .map_err(|e| anyhow::anyhow!("serialize verifier circuit data: {e:?}"))?;
    std::fs::write(&verifier_data_output_path, verifier_data_bytes)
        .with_context(|| format!("write {}", verifier_data_output_path.display()))?;
    let proof_manifest_json = serde_json::json!({
        "weights": opt.weights.display().to_string(),
        "input_image": opt.input_image.as_ref().map(|p| p.display().to_string()),
        "value_bits": opt.value_bits,
        "demo_fri": opt.demo_fri,
        "l1_low_memory_fri": opt.l1_low_memory_fri,
        "dimension": sums.len(),
        "h": exp_meta.h,
        "w": exp_meta.w,
        "activation_q": exp_meta.activation_q(),
        "quantize_q": exp_meta.quantize_q,
        "gap_cells": exp_meta.gap_cells(),
        "proof_path": proof_output_path.display().to_string(),
        "verifier_data_path": verifier_data_output_path.display().to_string(),
        "public_gap_sums": &sums,
    });
    std::fs::write(
        &manifest_output_path,
        serde_json::to_string_pretty(&proof_manifest_json)?,
    )
    .with_context(|| format!("write {}", manifest_output_path.display()))?;
    diag(&format!(
        "OK: final proof verifies. Public {}-D GAP sums: {:?}",
        sums.len(),
        sums
    ));
    diag(&format!(
        "Proof manifest saved to {}",
        manifest_output_path.display()
    ));
    Ok(())
}
