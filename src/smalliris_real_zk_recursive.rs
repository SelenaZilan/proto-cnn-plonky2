//! Recursive Plonky2 proving for the quantized SmallIris feature extractor.
//!
//! The current implementation uses a three-stage decomposition:
//! - **L1**: input image -> layer-1 public state `p1`
//! - **R2**: verify `L1` and apply layer 2 -> public state `p2`
//! - **R3**: verify `R2`, apply layer 3, and expose the final GAP sums
//!
//! This staging keeps each `build` / `prove` pass focused on one outer circuit.
//! In practice, peak memory is therefore closer to the largest recursive stage
//! than to a monolithic end-to-end construction.

use anyhow::Result;
use plonky2::{
    iop::{
        target::Target,
        witness::{PartialWitness, WitnessWrite},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{
            CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget,
            VerifierOnlyCircuitData,
        },
        config::{GenericConfig, PoseidonGoldilocksConfig},
        proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget},
    },
};

use crate::smalliris_real_zk::{
    conv_layer, field_to_i32_generic, forward_circuit_feature_sums, idx3, pool2, relu,
    rescale_tensor_nonnegative, to_field_i32, RealSmallIrisExport,
};

/// Configuration options for staged proving.
#[derive(Clone, Copy, Default)]
pub struct StagedProveConfig {
    /// For `L1` only, use slightly lighter FRI settings than the default
    /// recursion configuration. This offers marginal prover-side savings while
    /// remaining within Plonky2's configuration checks.
    pub l1_low_memory_fri: bool,
    /// Lower FRI security across all stages to accelerate local experiments and
    /// course-project runs. This is not intended as a production-security
    /// setting.
    pub demo_fri:          bool,
}

/// Slightly lighter FRI than the default recursion configuration while still
/// satisfying Plonky2 `check_config`:
/// `num_query_rounds * rate_bits + proof_of_work_bits >= security_bits`.
fn leaf_circuit_config_low_memory() -> CircuitConfig {
    let mut c = CircuitConfig::standard_recursion_config();
    c.security_bits = 80;
    c.fri_config.num_query_rounds = 24;
    c.fri_config.proof_of_work_bits = 16; // 24*3 + 16 = 88 >= 80
    c
}

/// Lower-security FRI settings for fast local experimentation.
fn demo_course_recursion_config() -> CircuitConfig {
    let mut c = CircuitConfig::standard_recursion_config();
    c.security_bits = 64;
    c.fri_config.num_query_rounds = 18;
    c.fri_config.proof_of_work_bits = 16; // 18*3 + 16 = 70 >= 64
    c
}

fn circuit_config_for_l1(staged: StagedProveConfig) -> CircuitConfig {
    if staged.demo_fri {
        demo_course_recursion_config()
    } else if staged.l1_low_memory_fri {
        leaf_circuit_config_low_memory()
    } else {
        CircuitConfig::standard_recursion_config()
    }
}

/// Choose the outer recursive-stage configuration. By default, R2 and R3 use
/// the standard recursion security settings; `demo_fri` lowers all stages.
fn circuit_config_for_recursive_stage(staged: StagedProveConfig) -> CircuitConfig {
    if staged.demo_fri {
        demo_course_recursion_config()
    } else {
        CircuitConfig::standard_recursion_config()
    }
}

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type Builder = CircuitBuilder<F, D>;

pub type RealSmallIrisRecursiveProof = ProofWithPublicInputs<F, C, D>;

/// Log to stderr and flush so lines are visible if the process is killed (e.g.
/// OOM) right after.
macro_rules! diag {
    ($($t:tt)*) => {{
        eprintln!($($t)*);
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }};
}

fn register_chw_public(builder: &mut Builder, t: &[Target], cin: usize, h: usize, w: usize) {
    debug_assert_eq!(t.len(), cin * h * w);
    for c in 0..cin {
        for i in 0..h {
            for j in 0..w {
                builder.register_public_input(t[idx3(c, i, j, h, w)]);
            }
        }
    }
}

fn num_public_layer1_total(export: &RealSmallIrisExport) -> usize {
    export.num_public_layer1()
}

fn num_public_layer2_total(export: &RealSmallIrisExport) -> usize {
    export.num_public_layer2()
}

fn log_public_mismatch(stage: &str, got: &[F], expected: &[i32]) {
    if got.len() != expected.len() {
        diag!(
            "{} mismatch: public_inputs len {} != expected {}",
            stage,
            got.len(),
            expected.len()
        );
        return;
    }
    if let Some((idx, (&x, &e))) = got
        .iter()
        .zip(expected.iter())
        .enumerate()
        .find(|(_, (&x, &e))| field_to_i32_generic(x) != e)
    {
        let got_i32 = field_to_i32_generic(x);
        let start = idx.saturating_sub(2);
        let end = (idx + 3).min(expected.len());
        let got_window: Vec<i32> = got[start..end]
            .iter()
            .map(|&v| field_to_i32_generic(v))
            .collect();
        let exp_window = &expected[start..end];
        diag!(
            "{} mismatch at idx {}: proof={} expected={} | window proof={:?} expected={:?}",
            stage,
            idx,
            got_i32,
            e,
            got_window,
            exp_window
        );
    } else {
        diag!("{} matches integer reference.", stage);
    }
}

// --- Stage 1: image → p1 ---

pub struct RealSmallIrisLayer1Circuit {
    pub input_targets: Vec<Target>,
    pub data:          CircuitData<F, C, D>,
    /// Expected `input_i32.len()` for [`Self::prove`].
    pub input_len:     usize,
}

impl RealSmallIrisLayer1Circuit {
    pub fn build(
        export: &RealSmallIrisExport,
        value_bits: usize,
        staged: StagedProveConfig,
    ) -> Result<Self> {
        export.validate()?;
        let config = circuit_config_for_l1(staged);
        if staged.demo_fri {
            diag!(
                "smalliris_recursive [1/3]: L1 using --demo-fri (FRI ~64-bit nominal; demos only)."
            );
        } else if staged.l1_low_memory_fri {
            diag!(
                "smalliris_recursive [1/3]: L1 using lighter FRI (~80-bit nominal vs 100-bit \
                 default)."
            );
        }
        diag!(
            "smalliris_recursive [1/3]: layer-1 circuit ({}×{}, {} public p1)…",
            export.h,
            export.w,
            export.num_public_layer1()
        );
        let mut builder = Builder::new(config);
        let h = export.h;
        let w = export.w;
        let (h1, w1) = export.hw_after_pools(1);
        let q = export.quantize_q_i32();

        let mut input_targets = Vec::with_capacity(export.input_len());
        for _ in 0..export.input_len() {
            input_targets.push(builder.add_virtual_target());
        }
        let pre1 = conv_layer(
            &mut builder,
            &input_targets,
            3,
            h,
            w,
            &export.w1,
            &export.b1,
            export.c1,
            3,
            1,
        );
        let mut a1 = vec![builder.zero(); pre1.len()];
        for i in 0..pre1.len() {
            a1[i] = relu(&mut builder, pre1[i], value_bits);
        }
        let p1_pool = pool2(&mut builder, &a1, export.c1, h, w, value_bits);
        let p1 = rescale_tensor_nonnegative(&mut builder, &p1_pool, q);
        register_chw_public(&mut builder, &p1, export.c1, h1, w1);

        builder.print_gate_counts(0);
        let data = builder.build::<C>();
        diag!("smalliris_recursive [1/3]: layer-1 build done.");
        Ok(Self {
            input_targets,
            data,
            input_len: export.input_len(),
        })
    }

    pub fn prove(&self, input_i32: &[i32]) -> Result<RealSmallIrisRecursiveProof> {
        assert_eq!(input_i32.len(), self.input_len);
        let mut pw = PartialWitness::new();
        for (&t, &v) in self.input_targets.iter().zip(input_i32) {
            pw.set_target(t, to_field_i32(v))
                .map_err(|e| anyhow::anyhow!("pw L1: {e:?}"))?;
        }
        self.data
            .prove(pw)
            .map_err(|e| anyhow::anyhow!("prove L1: {e:?}"))
    }
}

// --- Stage 2: verify L1 + conv2 → p2 ---

pub struct RealSmallIrisRecursionL2Circuit {
    pub data:                  CircuitData<F, C, D>,
    pub proof_with_pis_target: ProofWithPublicInputsTarget<D>,
    pub inner_verifier_target: VerifierCircuitTarget,
}

impl RealSmallIrisRecursionL2Circuit {
    pub fn build(
        inner_common: &CommonCircuitData<F, D>,
        export: &RealSmallIrisExport,
        value_bits: usize,
        staged: StagedProveConfig,
    ) -> Result<Self> {
        export.validate()?;
        let n1 = export.num_public_layer1();
        let n1_total = num_public_layer1_total(export);
        anyhow::ensure!(
            inner_common.num_public_inputs == n1_total,
            "inner L1 num_public_inputs {} != export total public inputs {}",
            inner_common.num_public_inputs,
            n1_total
        );
        let (h1, w1) = export.hw_after_pools(1);
        let (h2, w2) = export.hw_after_pools(2);
        let q = export.quantize_q_i32();
        diag!(
            "smalliris_recursive [2/3]: R2 = verify L1 + layer2 ({}×{}, {} public p2)…",
            h2,
            w2,
            export.num_public_layer2()
        );
        let config = circuit_config_for_recursive_stage(staged);
        if staged.demo_fri {
            diag!(
                "smalliris_recursive [2/3]: R2 using --demo-fri (FRI ~64-bit nominal; demos only)."
            );
        }
        let mut builder = Builder::new(config);

        let pt = builder.add_virtual_proof_with_pis(inner_common);
        let inner_verifier_target =
            builder.add_virtual_verifier_data(inner_common.config.fri_config.cap_height);
        builder.verify_proof::<C>(&pt, &inner_verifier_target, inner_common);

        let p1: Vec<Target> = pt.public_inputs.clone();
        anyhow::ensure!(p1.len() == n1);

        let pre2 = conv_layer(
            &mut builder,
            &p1,
            export.c1,
            h1,
            w1,
            &export.w2,
            &export.b2,
            export.c2,
            3,
            1,
        );
        let mut a2 = vec![builder.zero(); pre2.len()];
        for i in 0..pre2.len() {
            a2[i] = relu(&mut builder, pre2[i], value_bits);
        }
        let p2_pool = pool2(&mut builder, &a2, export.c2, h1, w1, value_bits);
        let p2 = rescale_tensor_nonnegative(&mut builder, &p2_pool, q);
        register_chw_public(&mut builder, &p2, export.c2, h2, w2);

        builder.print_gate_counts(0);
        let data = builder.build::<C>();
        diag!("smalliris_recursive [2/3]: R2 build done.");
        Ok(Self {
            data,
            proof_with_pis_target: pt,
            inner_verifier_target,
        })
    }

    pub fn prove(
        &self,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
        inner_vd: &VerifierOnlyCircuitData<C, D>,
    ) -> Result<RealSmallIrisRecursiveProof> {
        diag!("smalliris_recursive [2/3]: R2 proving…");
        let mut pw = PartialWitness::new();
        pw.set_proof_with_pis_target(&self.proof_with_pis_target, inner_proof)
            .map_err(|e| anyhow::anyhow!("pw R2 proof: {e:?}"))?;
        pw.set_verifier_data_target(&self.inner_verifier_target, inner_vd)
            .map_err(|e| anyhow::anyhow!("pw R2 vd: {e:?}"))?;
        let out = self
            .data
            .prove(pw)
            .map_err(|e| anyhow::anyhow!("prove R2: {e:?}"))?;
        diag!("smalliris_recursive [2/3]: R2 prove done.");
        Ok(out)
    }
}

// --- Stage 3: verify R2 + conv3 + GAP ---

pub struct RealSmallIrisRecursionL3Circuit {
    pub data:                  CircuitData<F, C, D>,
    pub proof_with_pis_target: ProofWithPublicInputsTarget<D>,
    pub inner_verifier_target: VerifierCircuitTarget,
}

impl RealSmallIrisRecursionL3Circuit {
    pub fn build(
        inner_common: &CommonCircuitData<F, D>,
        export: &RealSmallIrisExport,
        value_bits: usize,
        staged: StagedProveConfig,
    ) -> Result<Self> {
        export.validate()?;
        let n2 = export.num_public_layer2();
        let n2_total = num_public_layer2_total(export);
        anyhow::ensure!(
            inner_common.num_public_inputs == n2_total,
            "inner R2 num_public_inputs {} != export total public inputs {}",
            inner_common.num_public_inputs,
            n2_total
        );
        let (h2, w2) = export.hw_after_pools(2);
        let (hh, ww) = export.hw_after_pools(3);
        let q = export.quantize_q_i32();
        diag!(
            "smalliris_recursive [3/3]: R3 = verify R2 + layer3 + GAP ({} public, {}×{} cells)…",
            export.num_public_layer3(),
            hh,
            ww
        );
        let config = circuit_config_for_recursive_stage(staged);
        if staged.demo_fri {
            diag!(
                "smalliris_recursive [3/3]: R3 using --demo-fri (FRI ~64-bit nominal; demos only)."
            );
        }
        let mut builder = Builder::new(config);

        let pt = builder.add_virtual_proof_with_pis(inner_common);
        let inner_verifier_target =
            builder.add_virtual_verifier_data(inner_common.config.fri_config.cap_height);

        let p2: Vec<Target> = pt.public_inputs.clone();
        anyhow::ensure!(p2.len() == n2);

        // Build CNN tail first, then inner verification: changes generator / gate order
        // and avoids some witness-scheduling clashes observed only on very
        // large R3 graphs.
        let pre3 = conv_layer(
            &mut builder,
            &p2,
            export.c2,
            h2,
            w2,
            &export.w3,
            &export.b3,
            export.c3,
            3,
            1,
        );
        let mut a3 = vec![builder.zero(); pre3.len()];
        for i in 0..pre3.len() {
            a3[i] = relu(&mut builder, pre3[i], value_bits);
        }
        let p3_pool = pool2(&mut builder, &a3, export.c3, h2, w2, value_bits);
        let p3 = rescale_tensor_nonnegative(&mut builder, &p3_pool, q);

        for c in 0..export.c3 {
            let mut s = builder.zero();
            for i in 0..hh {
                for j in 0..ww {
                    s = builder.add(s, p3[idx3(c, i, j, hh, ww)]);
                }
            }
            builder.register_public_input(s);
        }

        builder.verify_proof::<C>(&pt, &inner_verifier_target, inner_common);

        builder.print_gate_counts(0);
        let data = builder.build::<C>();
        diag!("smalliris_recursive [3/3]: R3 build done.");
        Ok(Self {
            data,
            proof_with_pis_target: pt,
            inner_verifier_target,
        })
    }

    pub fn prove(
        &self,
        inner_proof: &ProofWithPublicInputs<F, C, D>,
        inner_vd: &VerifierOnlyCircuitData<C, D>,
    ) -> Result<RealSmallIrisRecursiveProof> {
        diag!("smalliris_recursive [3/3]: R3 proving…");
        let mut pw = PartialWitness::new();
        pw.set_proof_with_pis_target(&self.proof_with_pis_target, inner_proof)
            .map_err(|e| anyhow::anyhow!("pw R3 proof: {e:?}"))?;
        pw.set_verifier_data_target(&self.inner_verifier_target, inner_vd)
            .map_err(|e| anyhow::anyhow!("pw R3 vd: {e:?}"))?;
        let out = self
            .data
            .prove(pw)
            .map_err(|e| anyhow::anyhow!("prove R3: {e:?}"))?;
        diag!("smalliris_recursive [3/3]: R3 prove done.");
        Ok(out)
    }
}

pub fn verify_final(
    data: &CircuitData<F, C, D>,
    proof: &RealSmallIrisRecursiveProof,
) -> Result<()> {
    data.verify(proof.clone())
        .map_err(|e| anyhow::anyhow!("verify final: {e:?}"))
}

pub fn public_gap_sums_from_final_proof(proof: &RealSmallIrisRecursiveProof) -> Vec<i32> {
    proof.public_inputs
        .iter()
        .map(|&x| field_to_i32_generic(x))
        .collect()
}

pub fn public_gap_sums_from_final_proof_with_dimension(
    proof: &RealSmallIrisRecursiveProof,
    gap_dimension: usize,
) -> Vec<i32> {
    proof.public_inputs[..gap_dimension.min(proof.public_inputs.len())]
        .iter()
        .map(|&x| field_to_i32_generic(x))
        .collect()
}

/// Only stage 1: build + prove + verify. Use to see whether the machine
/// survives **L1** (still large).
pub fn smoke_layer1_only(
    export: &RealSmallIrisExport,
    value_bits: usize,
    input_i32: &[i32],
    staged: StagedProveConfig,
) -> Result<()> {
    diag!("smoke_layer1: building L1 circuit…");
    let layer1 = RealSmallIrisLayer1Circuit::build(export, value_bits, staged)?;
    diag!("smoke_layer1: proving…");
    let proof = layer1.prove(input_i32)?;
    diag!("smoke_layer1: verifying proof…");
    layer1
        .data
        .verify(proof)
        .map_err(|e| anyhow::anyhow!("verify L1: {e:?}"))?;
    diag!("smoke_layer1: OK (full 3-stage pipeline was NOT run).");
    Ok(())
}

/// Three proofs: L1 → R2 (verify L1 + conv2) → R3 (verify R2 + conv3 + GAP).
/// Drops each stage’s `CircuitData` after cloning `common` / `verifier_only`
/// for the next stage.
pub fn prove_three_stage(
    export: RealSmallIrisExport,
    value_bits: usize,
    input_i32: &[i32],
    staged: StagedProveConfig,
) -> Result<(RealSmallIrisRecursiveProof, CircuitData<F, C, D>, Vec<i32>)> {
    let sums_expected = forward_circuit_feature_sums(&export, input_i32);

    diag!("prove_three_stage: stage L1 — building…");
    let layer1 = RealSmallIrisLayer1Circuit::build(&export, value_bits, staged)?;
    diag!("prove_three_stage: stage L1 — proving…");
    let proof1 = layer1.prove(input_i32)?;
    diag!("prove_three_stage: stage L1 — done.");

    let p1_ref = crate::smalliris_real_zk::forward_int_after_layer1(&export, input_i32);
    log_public_mismatch("L1 public p1", &proof1.public_inputs, &p1_ref);

    let common1 = layer1.data.common.clone();
    let vd1 = layer1.data.verifier_only.clone();
    drop(layer1);

    diag!("prove_three_stage: stage R2 — building…");
    let r2 = RealSmallIrisRecursionL2Circuit::build(&common1, &export, value_bits, staged)?;
    diag!("prove_three_stage: stage R2 — proving…");
    let proof2 = r2.prove(&proof1, &vd1)?;
    diag!("prove_three_stage: stage R2 — done.");

    let p2_ref = crate::smalliris_real_zk::forward_int_after_layer2(&export, input_i32);
    log_public_mismatch("R2 public p2", &proof2.public_inputs, &p2_ref);

    let common2 = r2.data.common.clone();
    let vd2 = r2.data.verifier_only.clone();
    drop(r2);

    diag!("prove_three_stage: stage R3 — building…");
    let r3 = RealSmallIrisRecursionL3Circuit::build(&common2, &export, value_bits, staged)?;
    diag!("prove_three_stage: stage R3 — proving…");
    let proof_final = r3.prove(&proof2, &vd2)?;
    log_public_mismatch(
        "R3 public GAP sums",
        &proof_final.public_inputs,
        &sums_expected,
    );
    diag!("prove_three_stage: all stages done.");
    Ok((proof_final, r3.data, sums_expected))
}

/// Backwards-compatible name: now runs **three** stages (L1, R2, R3).
pub fn prove_two_stage(
    export: RealSmallIrisExport,
    value_bits: usize,
    input_i32: &[i32],
) -> Result<(RealSmallIrisRecursiveProof, CircuitData<F, C, D>, Vec<i32>)> {
    prove_three_stage(export, value_bits, input_i32, StagedProveConfig::default())
}

pub fn public_sums_match_final(proof: &RealSmallIrisRecursiveProof, expected: &[i32]) -> bool {
    proof.public_inputs.len() == expected.len()
        && proof
            .public_inputs
            .iter()
            .zip(expected)
            .all(|(&x, &e)| field_to_i32_generic(x) == e)
}
