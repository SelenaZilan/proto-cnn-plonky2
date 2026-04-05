//! Standalone verifier for the recursive SmallIris proof artifact.
//!
//! This binary:
//! - reads a serialized recursive proof from disk
//! - reads the corresponding verifier-side circuit data
//! - runs Plonky2 verification
//! - prints the verified public GAP sums as JSON
//!
//! Example:
//!
//! ```bash
//! cargo +nightly run --release --bin zk-smalliris-real-recursive-verify -- \
//!   --proof-path "./fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64_S5750L01_recursive_proof.bin" \
//!   --verifier-data-path "./fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64_S5750L01_recursive_verifier_data.bin"
//! ```

use anyhow::Context;
use cnn_zkp::{
    smalliris_real_zk_recursive::{
        public_gap_sums_from_final_proof_with_dimension, RealSmallIrisRecursiveProof,
    },
};
use plonky2::{
    field::extension::Extendable,
    hash::hash_types::RichField,
    plonk::{
        circuit_data::VerifierCircuitData,
        config::{GenericConfig, PoseidonGoldilocksConfig},
    },
    util::serialization::DefaultGateSerializer,
};
use std::path::PathBuf;
use structopt::StructOpt;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

#[derive(StructOpt)]
struct Opt {
    #[structopt(long, parse(from_os_str))]
    proof_path:         PathBuf,
    #[structopt(long, parse(from_os_str))]
    verifier_data_path: PathBuf,
    #[structopt(long, parse(from_os_str))]
    output_json:        Option<PathBuf>,
    #[structopt(long)]
    gap_dimension: Option<usize>,
}

fn load_verifier_data(path: &PathBuf) -> anyhow::Result<VerifierCircuitData<F, C, D>>
where
    F: RichField + Extendable<D>,
{
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    VerifierCircuitData::from_bytes(bytes, &DefaultGateSerializer)
        .map_err(|e| anyhow::anyhow!("decode verifier data {}: {e:?}", path.display()))
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let verifier_data = load_verifier_data(&opt.verifier_data_path)?;
    let proof_bytes = std::fs::read(&opt.proof_path)
        .with_context(|| format!("read {}", opt.proof_path.display()))?;
    let proof = RealSmallIrisRecursiveProof::from_bytes(proof_bytes, &verifier_data.common)
        .map_err(|e| anyhow::anyhow!("decode proof {}: {e:?}", opt.proof_path.display()))?;
    verifier_data
        .verify(proof.clone())
        .with_context(|| "verify recursive proof")?;

    let gap_dimension = opt
        .gap_dimension
        .unwrap_or(proof.public_inputs.len());
    let public_gap_sums = public_gap_sums_from_final_proof_with_dimension(&proof, gap_dimension);

    let sums_text = public_gap_sums
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        concat!(
            "{{\n",
            "  \"proof_path\": \"{}\",\n",
            "  \"verifier_data_path\": \"{}\",\n",
            "  \"proof_verified\": true,\n",
            "  \"dimension\": {},\n",
            "  \"public_gap_sums\": [{}],\n",
            "  \"gap_dimension_used\": {}\n",
            "}}"
        ),
        opt.proof_path.display(),
        opt.verifier_data_path.display(),
        public_gap_sums.len(),
        sums_text,
        gap_dimension
    );
    println!("{text}");

    if let Some(path) = opt.output_json {
        std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    }

    Ok(())
}
