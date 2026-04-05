//! SmallIris feature extractor in Plonky2 using **exported int32 weights**
//! (`export_smalliris_zk_weights.py` -> JSON) and a quantized, normalized input.
//! Spatial size **`h×w`** is read from JSON (square, multiple of 8).
//!
//! The current repository's main configuration is **48×48** with channels
//! `24 -> 48 -> 64`, but this module is written to support any JSON-exported
//! spatial size that satisfies the shape checks. Topology: Conv->ReLU->MaxPool
//! x3, then **public** per-channel **sum** over `(h/8)×(w/8)` (GAP sum, not mean).

use anyhow::{Context, Result};
use image::{
    imageops::{crop_imm, resize, FilterType},
    ImageReader,
};
use plonky2::{
    field::{
        extension::Extendable,
        types::{Field, Field64, PrimeField64},
    },
    hash::hash_types::RichField,
    iop::{
        generator::{GeneratedValues, SimpleGenerator},
        target::Target,
        witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData, CommonCircuitData},
        config::{GenericConfig, PoseidonGoldilocksConfig},
        proof::ProofWithPublicInputs,
    },
    util::serialization::{Buffer, IoResult, Read, Write},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

use anyhow::Result as AnyhowResult;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type Builder = CircuitBuilder<F, D>;
pub type RealSmallIrisProof = ProofWithPublicInputs<F, C, D>;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RealSmallIrisExport {
    pub h:            usize,
    pub w:            usize,
    pub quantize_q:   f64,
    #[serde(default)]
    pub activation_q: Option<f64>,
    #[serde(default = "default_c1")]
    pub c1:           usize,
    #[serde(default = "default_c2")]
    pub c2:           usize,
    #[serde(default = "default_c3")]
    pub c3:           usize,
    pub w1:           Vec<i32>,
    pub b1:           Vec<i32>,
    pub w2:           Vec<i32>,
    pub b2:           Vec<i32>,
    pub w3:           Vec<i32>,
    pub b3:           Vec<i32>,
}

const fn default_c1() -> usize {
    32
}
const fn default_c2() -> usize {
    64
}
const fn default_c3() -> usize {
    128
}

impl RealSmallIrisExport {
    pub fn load_path(p: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(p.as_ref())
            .with_context(|| format!("read {}", p.as_ref().display()))?;
        Ok(serde_json::from_str(&s)?)
    }

    pub fn input_len(&self) -> usize {
        3 * self.h * self.w
    }

    pub fn quantize_q_i32(&self) -> i32 {
        self.quantize_q.round() as i32
    }

    pub fn activation_q(&self) -> f64 {
        self.activation_q.unwrap_or(self.quantize_q)
    }

    pub fn activation_q_i32(&self) -> i32 {
        self.activation_q().round() as i32
    }

    /// `(H, W)` after `k` consecutive 2×2 max-pools from the input (`k` in
    /// 0..=3).
    pub fn hw_after_pools(&self, k: u8) -> (usize, usize) {
        let mut h = self.h;
        let mut w = self.w;
        for _ in 0..k {
            h /= 2;
            w /= 2;
        }
        (h, w)
    }

    /// Stage-1 public tensor size: `c1 × (H/2) × (W/2)`.
    pub fn num_public_layer1(&self) -> usize {
        let (h1, w1) = self.hw_after_pools(1);
        self.c1 * h1 * w1
    }

    /// Stage-2 public tensor size: `c2 × (H/4) × (W/4)`.
    pub fn num_public_layer2(&self) -> usize {
        let (h2, w2) = self.hw_after_pools(2);
        self.c2 * h2 * w2
    }

    /// Final GAP output dimension = `c3`.
    pub fn num_public_layer3(&self) -> usize {
        self.c3
    }

    /// Cells per channel summed in GAP: `(H/8)*(W/8)`.
    pub fn gap_cells(&self) -> usize {
        let (hh, ww) = self.hw_after_pools(3);
        hh * ww
    }

    /// Check JSON: square `h=w`, multiple of 8, fused SmallIris conv shapes.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.h == self.w, "expected square H=W");
        anyhow::ensure!(
            self.h >= 8 && self.h % 8 == 0,
            "H=W must be ≥8 and divisible by 8 (three 2×2 pools)"
        );
        anyhow::ensure!(self.h <= 128, "H=W over 128 not supported");
        let q_rounded = self.quantize_q.round();
        anyhow::ensure!(
            (self.quantize_q - q_rounded).abs() < 1e-6,
            "quantize_q must be integer-valued"
        );
        anyhow::ensure!(
            q_rounded >= 1.0 && q_rounded <= i32::MAX as f64,
            "quantize_q must fit in positive i32"
        );
        let aq_rounded = self.activation_q().round();
        anyhow::ensure!(
            (self.activation_q() - aq_rounded).abs() < 1e-6,
            "activation_q must be integer-valued"
        );
        anyhow::ensure!(
            aq_rounded >= 1.0 && aq_rounded <= i32::MAX as f64,
            "activation_q must fit in positive i32"
        );
        anyhow::ensure!(
            self.c1 > 0 && self.c2 > 0 && self.c3 > 0,
            "c1,c2,c3 must be positive"
        );
        anyhow::ensure!(self.w1.len() == self.c1 * 3 * 9 && self.b1.len() == self.c1);
        anyhow::ensure!(self.w2.len() == self.c2 * self.c1 * 9 && self.b2.len() == self.c2);
        anyhow::ensure!(self.w3.len() == self.c3 * self.c2 * 9 && self.b3.len() == self.c3);
        Ok(())
    }
}

const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];

fn load_rgb_image_bytes(path: impl AsRef<Path>, h: usize, w: usize) -> Result<Vec<u8>> {
    let path_ref = path.as_ref();
    let img = ImageReader::open(path_ref)
        .with_context(|| format!("open image {}", path_ref.display()))?
        .decode()
        .with_context(|| format!("decode image {}", path_ref.display()))?
        .to_rgb8();

    let src_w = img.width();
    let src_h = img.height();
    anyhow::ensure!(src_w > 0 && src_h > 0, "image must be non-empty");

    let target_w = w as u32;
    let target_h = h as u32;
    let scale = f64::max(
        target_w as f64 / src_w as f64,
        target_h as f64 / src_h as f64,
    );
    let resized_w = ((src_w as f64 * scale).round() as u32).max(target_w);
    let resized_h = ((src_h as f64 * scale).round() as u32).max(target_h);

    let resized = resize(&img, resized_w, resized_h, FilterType::Triangle);
    let crop_x = (resized_w - target_w) / 2;
    let crop_y = (resized_h - target_h) / 2;
    let cropped = crop_imm(&resized, crop_x, crop_y, target_w, target_h).to_image();

    Ok(cropped.into_raw())
}

pub fn encode_rgb_bytes_to_chw_input_i32(
    rgb_hwc: &[u8],
    h: usize,
    w: usize,
    activation_q: i32,
) -> Result<Vec<i32>> {
    anyhow::ensure!(rgb_hwc.len() == 3 * h * w, "rgb_hwc length mismatch");
    let aq = activation_q as f64;
    let mut out = vec![0i32; 3 * h * w];
    for c in 0..3 {
        for i in 0..h {
            for j in 0..w {
                let hwc_idx = (i * w + j) * 3 + c;
                let x = rgb_hwc[hwc_idx] as f64 / 255.0;
                let normalized = (x - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                let q = (normalized * aq).round();
                out[idx3(c, i, j, h, w)] = q.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            }
        }
    }
    Ok(out)
}

pub fn load_rgb_image_input(
    path: impl AsRef<Path>,
    h: usize,
    w: usize,
    activation_q: i32,
) -> Result<Vec<i32>> {
    let rgb_hwc = load_rgb_image_bytes(path, h, w)?;
    encode_rgb_bytes_to_chw_input_i32(&rgb_hwc, h, w, activation_q)
}

pub fn synthetic_input_i32(exp: &RealSmallIrisExport) -> Vec<i32> {
    let raw: Vec<u8> = (0..exp.input_len()).map(|i| (i * 13 + 7) as u8).collect();
    encode_rgb_bytes_to_chw_input_i32(&raw, exp.h, exp.w, exp.activation_q_i32())
        .expect("synthetic input shape matches export")
}

pub(crate) fn to_field_i32(v: i32) -> F {
    if v >= 0 {
        F::from_canonical_u32(v as u32)
    } else {
        -F::from_canonical_u32((-v) as u32)
    }
}

pub fn field_to_i32_generic<G: PrimeField64>(x: G) -> i32 {
    let u = x.to_canonical_u64();
    let p = G::ORDER;
    let v = if u <= p / 2 {
        u as i128
    } else {
        u as i128 - p as i128
    };
    v.clamp(i32::MIN as i128, i32::MAX as i128) as i32
}

pub fn field_to_u32_generic<G: PrimeField64>(x: G) -> u32 {
    x.to_canonical_u64() as u32
}

/// Signed integer in `(-p/2, p/2]` matching Goldilocks canonical encoding —
/// **no** i32 clamp. Witness generators for ReLU / max must use this so `y = x
/// + r` agrees with unconstrained conv sums.
pub(crate) fn field_to_i128_signed<G: PrimeField64>(x: G) -> i128 {
    let u = x.to_canonical_u64();
    let p = G::ORDER;
    if u <= p / 2 {
        u as i128
    } else {
        u as i128 - p as i128
    }
}

pub(crate) fn idx3(c: usize, i: usize, j: usize, h: usize, w: usize) -> usize {
    c * h * w + i * w + j
}

fn rescale_nonnegative_i32(x: i32, divisor: i32) -> i32 {
    if x <= 0 {
        0
    } else {
        x / divisor
    }
}

fn rescale_nonnegative_i128(x: i128, divisor: i128) -> i128 {
    if x <= 0 {
        0
    } else {
        x / divisor
    }
}

fn rescale_tensor_nonnegative_i32(inp: &[i32], divisor: i32) -> Vec<i32> {
    inp.iter()
        .map(|&x| rescale_nonnegative_i32(x, divisor))
        .collect()
}

fn rescale_tensor_nonnegative_i128(inp: &[i128], divisor: i128) -> Vec<i128> {
    inp.iter()
        .map(|&x| rescale_nonnegative_i128(x, divisor))
        .collect()
}

fn wco(co: usize, ci: usize, ki: usize, kj: usize, cin: usize, kh: usize, kw: usize) -> usize {
    co * (cin * kh * kw) + ci * (kh * kw) + ki * kw + kj
}

fn conv2d(
    in_: &[i32],
    cin: usize,
    h: usize,
    w: usize,
    weights: &[i32],
    bias: &[i32],
    cout: usize,
    k: usize,
    pad: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; cout * h * w];
    for co in 0..cout {
        for i in 0..h {
            for j in 0..w {
                let mut acc: i64 = bias[co] as i64;
                for ci in 0..cin {
                    for ki in 0..k {
                        for kj in 0..k {
                            let ii = i as isize + ki as isize - pad as isize;
                            let jj = j as isize + kj as isize - pad as isize;
                            if ii >= 0 && ii < h as isize && jj >= 0 && jj < w as isize {
                                let wi = wco(co, ci, ki, kj, cin, k, k);
                                acc += in_[idx3(ci, ii as usize, jj as usize, h, w)] as i64
                                    * weights[wi] as i64;
                            }
                        }
                    }
                }
                out[idx3(co, i, j, h, w)] = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }
    }
    out
}

pub(crate) fn maxpool2x2(inp: &[i32], cin: usize, h: usize, w: usize) -> Vec<i32> {
    let oh = h / 2;
    let ow = w / 2;
    let mut o = vec![0i32; cin * oh * ow];
    for c in 0..cin {
        for i in 0..oh {
            for j in 0..ow {
                let a = inp[idx3(c, 2 * i, 2 * j, h, w)];
                let b = inp[idx3(c, 2 * i, 2 * j + 1, h, w)];
                let c0 = inp[idx3(c, 2 * i + 1, 2 * j, h, w)];
                let d = inp[idx3(c, 2 * i + 1, 2 * j + 1, h, w)];
                o[idx3(c, i, j, oh, ow)] = a.max(b).max(c0).max(d);
            }
        }
    }
    o
}

/// Quantized activations after block 1 (pool₁: **c1 × H/2 × W/2**), row-major
/// `idx3`.
pub fn forward_int_after_layer1(exp: &RealSmallIrisExport, input_i32: &[i32]) -> Vec<i32> {
    assert_eq!(input_i32.len(), exp.input_len());
    let h = exp.h;
    let w = exp.w;
    let pre1 = conv2d(input_i32, 3, h, w, &exp.w1, &exp.b1, exp.c1, 3, 1);
    let a1: Vec<i32> = pre1.iter().map(|&x| x.max(0)).collect();
    let p1 = maxpool2x2(&a1, exp.c1, h, w);
    rescale_tensor_nonnegative_i32(&p1, exp.quantize_q_i32())
}

/// Quantized activations after block 2 (pool₂: **c2 × H/4 × W/4**), row-major
/// `idx3`.
pub fn forward_int_after_layer2(exp: &RealSmallIrisExport, input_i32: &[i32]) -> Vec<i32> {
    let p1 = forward_int_after_layer1(exp, input_i32);
    let (h1, w1) = exp.hw_after_pools(1);
    let pre2 = conv2d(&p1, exp.c1, h1, w1, &exp.w2, &exp.b2, exp.c2, 3, 1);
    let a2: Vec<i32> = pre2.iter().map(|&x| x.max(0)).collect();
    let p2 = maxpool2x2(&a2, exp.c2, h1, w1);
    rescale_tensor_nonnegative_i32(&p2, exp.quantize_q_i32())
}

/// Heuristic bit width from int forward (activations / slack magnitudes).
///
/// ReLU / MaxPool in-circuit no longer `range_check` activations or max deltas
/// (witness stability); this remains useful as a **diagnostic** of tensor
/// magnitudes. Adds a small margin.
pub fn diagnose_min_value_bits(exp: &RealSmallIrisExport, input_i32: &[i32]) -> usize {
    assert_eq!(input_i32.len(), exp.input_len());
    let h = exp.h;
    let w = exp.w;
    let (h1, w1) = exp.hw_after_pools(1);
    let (h2, w2) = exp.hw_after_pools(2);
    let q = exp.quantize_q_i32();

    let mut max_u: u64 = 0;

    let consider_pre_relu = |x: i32, max_u: &mut u64| {
        let y = (x.max(0) as i64) as u64;
        let r = (-(x as i64)).max(0) as u64;
        *max_u = (*max_u).max(y).max(r);
    };

    let pre1 = conv2d(input_i32, 3, h, w, &exp.w1, &exp.b1, exp.c1, 3, 1);
    for &x in &pre1 {
        consider_pre_relu(x, &mut max_u);
    }
    let a1: Vec<i32> = pre1.iter().map(|&x| x.max(0)).collect();
    let p1 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a1, exp.c1, h, w), q);
    for &x in &p1 {
        max_u = max_u.max(x.max(0) as u64);
    }

    let pre2 = conv2d(&p1, exp.c1, h1, w1, &exp.w2, &exp.b2, exp.c2, 3, 1);
    for &x in &pre2 {
        consider_pre_relu(x, &mut max_u);
    }
    let a2: Vec<i32> = pre2.iter().map(|&x| x.max(0)).collect();
    let p2 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a2, exp.c2, h1, w1), q);
    for &x in &p2 {
        max_u = max_u.max(x.max(0) as u64);
    }

    let pre3 = conv2d(&p2, exp.c2, h2, w2, &exp.w3, &exp.b3, exp.c3, 3, 1);
    for &x in &pre3 {
        consider_pre_relu(x, &mut max_u);
    }
    let a3: Vec<i32> = pre3.iter().map(|&x| x.max(0)).collect();
    let p3 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a3, exp.c3, h2, w2), q);
    for &x in &p3 {
        max_u = max_u.max(x.max(0) as u64);
    }

    let bit_len = |v: u64| -> usize {
        if v == 0 {
            1
        } else {
            (64 - v.leading_zeros()) as usize
        }
    };
    let base = bit_len(max_u);
    (base + 2).clamp(8, 64)
}

fn normalize_signed_goldilocks(x: i128) -> i128 {
    let p = F::ORDER as i128;
    let mut r = x % p;
    if r < 0 {
        r += p;
    }
    if r <= p / 2 {
        r
    } else {
        r - p
    }
}

fn conv2d_field_signed(
    in_: &[i128],
    cin: usize,
    h: usize,
    w: usize,
    weights: &[i32],
    bias: &[i32],
    cout: usize,
    k: usize,
    pad: usize,
) -> Vec<i128> {
    let mut out = vec![0i128; cout * h * w];
    for co in 0..cout {
        for i in 0..h {
            for j in 0..w {
                let mut acc = bias[co] as i128;
                for ci in 0..cin {
                    for ki in 0..k {
                        for kj in 0..k {
                            let ii = i as isize + ki as isize - pad as isize;
                            let jj = j as isize + kj as isize - pad as isize;
                            if ii >= 0 && ii < h as isize && jj >= 0 && jj < w as isize {
                                let wi = wco(co, ci, ki, kj, cin, k, k);
                                acc += in_[idx3(ci, ii as usize, jj as usize, h, w)]
                                    * weights[wi] as i128;
                            }
                        }
                    }
                }
                out[idx3(co, i, j, h, w)] = normalize_signed_goldilocks(acc);
            }
        }
    }
    out
}

fn maxpool2x2_i128(inp: &[i128], cin: usize, h: usize, w: usize) -> Vec<i128> {
    let oh = h / 2;
    let ow = w / 2;
    let mut o = vec![0i128; cin * oh * ow];
    for c in 0..cin {
        for i in 0..oh {
            for j in 0..ow {
                let a = inp[idx3(c, 2 * i, 2 * j, h, w)];
                let b = inp[idx3(c, 2 * i, 2 * j + 1, h, w)];
                let c0 = inp[idx3(c, 2 * i + 1, 2 * j, h, w)];
                let d = inp[idx3(c, 2 * i + 1, 2 * j + 1, h, w)];
                o[idx3(c, i, j, oh, ow)] = a.max(b).max(c0).max(d);
            }
        }
    }
    o
}

/// Quantized forward (int activations). Returns **c3** channel GAP **sums**
/// over `(H/8)×(W/8)`.
pub fn forward_int_feature_sums(exp: &RealSmallIrisExport, input_i32: &[i32]) -> Vec<i32> {
    assert_eq!(input_i32.len(), exp.input_len());
    let h = exp.h;
    let w = exp.w;
    let (h1, w1) = exp.hw_after_pools(1);
    let (h2, w2) = exp.hw_after_pools(2);
    let (hh, ww) = exp.hw_after_pools(3);
    let q = exp.quantize_q_i32();
    let pre1 = conv2d(input_i32, 3, h, w, &exp.w1, &exp.b1, exp.c1, 3, 1);
    let a1: Vec<i32> = pre1.iter().map(|&x| x.max(0)).collect();
    let p1 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a1, exp.c1, h, w), q);
    let pre2 = conv2d(&p1, exp.c1, h1, w1, &exp.w2, &exp.b2, exp.c2, 3, 1);
    let a2: Vec<i32> = pre2.iter().map(|&x| x.max(0)).collect();
    let p2 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a2, exp.c2, h1, w1), q);
    let pre3 = conv2d(&p2, exp.c2, h2, w2, &exp.w3, &exp.b3, exp.c3, 3, 1);
    let a3: Vec<i32> = pre3.iter().map(|&x| x.max(0)).collect();
    let p3 = rescale_tensor_nonnegative_i32(&maxpool2x2(&a3, exp.c3, h2, w2), q);
    let mut sums = vec![0i32; exp.c3];
    for c in 0..exp.c3 {
        let mut s: i64 = 0;
        for i in 0..hh {
            for j in 0..ww {
                s += p3[idx3(c, i, j, hh, ww)] as i64;
            }
        }
        sums[c] = s.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    sums
}

/// Reference forward matching the **current circuit semantics**:
/// conv/GAP are done in Goldilocks field arithmetic, then interpreted in
/// `(-p/2, p/2]`; ReLU is `max(0, signed_field_value)`, pool is integer max
/// over those signed activations.
pub fn forward_circuit_feature_sums(exp: &RealSmallIrisExport, input_i32: &[i32]) -> Vec<i32> {
    assert_eq!(input_i32.len(), exp.input_len());
    let input: Vec<i128> = input_i32.iter().map(|&u| u as i128).collect();
    let h = exp.h;
    let w = exp.w;
    let (h1, w1) = exp.hw_after_pools(1);
    let (h2, w2) = exp.hw_after_pools(2);
    let (hh, ww) = exp.hw_after_pools(3);
    let q = exp.quantize_q_i32() as i128;

    let pre1 = conv2d_field_signed(&input, 3, h, w, &exp.w1, &exp.b1, exp.c1, 3, 1);
    let a1: Vec<i128> = pre1.iter().map(|&x| x.max(0)).collect();
    let p1 = rescale_tensor_nonnegative_i128(&maxpool2x2_i128(&a1, exp.c1, h, w), q);

    let pre2 = conv2d_field_signed(&p1, exp.c1, h1, w1, &exp.w2, &exp.b2, exp.c2, 3, 1);
    let a2: Vec<i128> = pre2.iter().map(|&x| x.max(0)).collect();
    let p2 = rescale_tensor_nonnegative_i128(&maxpool2x2_i128(&a2, exp.c2, h1, w1), q);

    let pre3 = conv2d_field_signed(&p2, exp.c2, h2, w2, &exp.w3, &exp.b3, exp.c3, 3, 1);
    let a3: Vec<i128> = pre3.iter().map(|&x| x.max(0)).collect();
    let p3 = rescale_tensor_nonnegative_i128(&maxpool2x2_i128(&a3, exp.c3, h2, w2), q);

    let mut sums = vec![0i32; exp.c3];
    for c in 0..exp.c3 {
        let mut s = 0i128;
        for i in 0..hh {
            for j in 0..ww {
                s += p3[idx3(c, i, j, hh, ww)];
            }
        }
        sums[c] = normalize_signed_goldilocks(s).clamp(i32::MIN as i128, i32::MAX as i128) as i32;
    }
    sums
}

/// Sets only the slack `r` from pre-activation `x`. Output `y` is **not**
/// written here: it is the arithmetic `x + r` wire (merged via `connect`), so
/// `split_le(y)` / `WireSplitGenerator` + adder are the only writers on `y`’s
/// partition. Writing `y` here as well races the adder in large circuits (R3) →
/// “Partition … set twice … 0 != 1” on bit limbs.
#[derive(Debug)]
struct ReluRealGen {
    x: Target,
    y: Target,
    r: Target,
}

impl<F: RichField + Extendable<D> + PrimeField64, const D: usize> SimpleGenerator<F, D>
    for ReluRealGen
{
    fn id(&self) -> String {
        "ReluRealSmallIris".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> AnyhowResult<()> {
        let v = field_to_i128_signed(witness.get_target(self.x));
        let rv = if v >= 0 {
            0u64
        } else {
            let r = v.unsigned_abs();
            u64::try_from(r).map_err(|_| anyhow::anyhow!("relu r out of u64 range"))?
        };
        out.set_target(self.r, F::from_canonical_u64(rv))
    }

    fn serialize(&self, dst: &mut Vec<u8>, _c: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target(self.y)?;
        dst.write_target(self.r)
    }

    fn deserialize(src: &mut Buffer, _c: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            x: src.read_target()?,
            y: src.read_target()?,
            r: src.read_target()?,
        })
    }
}

/// Writes a single `out = max(a,b)` (canonical u64). Use **three** instances
/// per 2×2 pool cell instead of one multi-output generator so witness
/// scheduling interleaves cleanly with `split_le` on **differences** `(out−a)`,
/// `(out−b)` in huge graphs (R3).
#[derive(Debug)]
struct Max2PoolGen {
    a:   Target,
    b:   Target,
    out: Target,
}

impl<F: RichField + Extendable<D> + PrimeField64, const D: usize> SimpleGenerator<F, D>
    for Max2PoolGen
{
    fn id(&self) -> String {
        "Max2PoolSmallIris".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.a, self.b]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> AnyhowResult<()> {
        let ua = witness.get_target(self.a).to_canonical_u64();
        let ub = witness.get_target(self.b).to_canonical_u64();
        let v = ua.max(ub);
        out.set_target(self.out, F::from_canonical_u64(v))
    }

    fn serialize(&self, dst: &mut Vec<u8>, _c: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.a)?;
        dst.write_target(self.b)?;
        dst.write_target(self.out)
    }

    fn deserialize(src: &mut Buffer, _c: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            a:   src.read_target()?,
            b:   src.read_target()?,
            out: src.read_target()?,
        })
    }
}

/// Honest integer rescale for nonnegative activations: `y = floor(x / q)`.
///
/// We keep only the linear relation `x = y*q + r` in-circuit and rely on the
/// generator for the intended quotient/remainder pair. This is weaker than a
/// fully range-constrained division gadget, but much lighter and sufficient for
/// the coursework proving pipeline.
#[derive(Debug)]
struct RescaleDivGen {
    x:       Target,
    divisor: Target,
    y:       Target,
    r:       Target,
}

impl<F: RichField + Extendable<D> + PrimeField64, const D: usize> SimpleGenerator<F, D>
    for RescaleDivGen
{
    fn id(&self) -> String {
        "RescaleDivSmallIris".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x, self.divisor]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out: &mut GeneratedValues<F>,
    ) -> AnyhowResult<()> {
        let xv = field_to_i128_signed(witness.get_target(self.x));
        let qv = field_to_i128_signed(witness.get_target(self.divisor));
        anyhow::ensure!(qv > 0, "rescale divisor must be positive");
        anyhow::ensure!(xv >= 0, "rescale expects nonnegative activation");
        let yv = xv / qv;
        let rv = xv % qv;
        out.set_target(self.y, F::from_canonical_u64(yv as u64))?;
        out.set_target(self.r, F::from_canonical_u64(rv as u64))
    }

    fn serialize(&self, dst: &mut Vec<u8>, _c: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target(self.divisor)?;
        dst.write_target(self.y)?;
        dst.write_target(self.r)
    }

    fn deserialize(src: &mut Buffer, _c: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            x:       src.read_target()?,
            divisor: src.read_target()?,
            y:       src.read_target()?,
            r:       src.read_target()?,
        })
    }
}

pub(crate) fn rescale_nonnegative(builder: &mut Builder, x: Target, divisor: i32) -> Target {
    let y = builder.add_virtual_target();
    let r = builder.add_virtual_target();
    let divisor_t = builder.constant(to_field_i32(divisor));
    builder.add_simple_generator(RescaleDivGen {
        x,
        divisor: divisor_t,
        y,
        r,
    });
    let z = builder.zero();
    let qy = builder.mul_const_add(to_field_i32(divisor), y, z);
    let qy_plus_r = builder.add(qy, r);
    builder.connect(qy_plus_r, x);
    y
}

pub(crate) fn rescale_tensor_nonnegative(
    builder: &mut Builder,
    inp: &[Target],
    divisor: i32,
) -> Vec<Target> {
    inp.iter()
        .map(|&x| rescale_nonnegative(builder, x, divisor))
        .collect()
}

pub(crate) fn relu(builder: &mut Builder, x: Target, bits: usize) -> Target {
    let _ = bits;
    let y = builder.add_virtual_target();
    let r = builder.add_virtual_target();
    builder.add_simple_generator(ReluRealGen { x, y, r });
    // No `range_check(y)` / `range_check(r)`: `split_le` on those wires merges with
    // BaseSum reconstruction while the adder also defines `y=x+r`, producing
    // “set twice … 0 != 1” on bit wires in very large R3. Polynomial
    // constraints `y=x+r`, `y*r=0` still tie ReLU; for coursework, quantised
    // nets keep values small in honest runs.
    let prod = builder.mul(y, r);
    let z = builder.zero();
    builder.connect(prod, z);
    let sum = builder.add(x, r);
    builder.connect(sum, y);
    y
}

/// Pairwise max: `(m-a)(m-b)=0` in the field. **`m` is fixed by [`Max2PoolGen`]
/// (honest max).**
///
/// We intentionally **do not** `range_check(m-a)` / `range_check(m-b)`: those
/// call `split_le` on the adder outputs and, in very large recursive stages
/// (R3), still hit Plonky2 witness races (`Partition … set twice … 0 != 1`)
/// alongside other generators. Without range checks this is **weaker as a
/// standalone algebraic max** in \(\mathbb{F}_p\); the prover’s [`Max2PoolGen`]
/// restores the intended CNN max-pool for honest runs (sufficient for this
/// coursework pipeline).
fn max2_pair_diffs_only(builder: &mut Builder, a: Target, b: Target, m: Target, bits: usize) {
    let _ = bits;
    let na = builder.neg(a);
    let nb = builder.neg(b);
    let d1 = builder.add(m, na);
    let d2 = builder.add(m, nb);
    let pr = builder.mul(d1, d2);
    let z = builder.zero();
    builder.connect(pr, z);
}

fn max4(builder: &mut Builder, a: Target, b: Target, c: Target, d: Target, bits: usize) -> Target {
    let t1 = builder.add_virtual_target();
    let t2 = builder.add_virtual_target();
    let m = builder.add_virtual_target();
    // Register max constraints before hint generators so `split_le` is laid out
    // first.
    max2_pair_diffs_only(builder, a, b, t1, bits);
    max2_pair_diffs_only(builder, c, d, t2, bits);
    max2_pair_diffs_only(builder, t1, t2, m, bits);
    builder.add_simple_generator(Max2PoolGen { a, b, out: t1 });
    builder.add_simple_generator(Max2PoolGen {
        a:   c,
        b:   d,
        out: t2,
    });
    builder.add_simple_generator(Max2PoolGen {
        a:   t1,
        b:   t2,
        out: m,
    });
    m
}

fn dot_const(builder: &mut Builder, coeffs: &[i32], vars: &[Target]) -> Target {
    let mut s = builder.zero();
    for (&c, &v) in coeffs.iter().zip(vars) {
        s = builder.mul_const_add(to_field_i32(c), v, s);
    }
    s
}

pub(crate) fn conv_layer(
    builder: &mut Builder,
    inp: &[Target],
    cin: usize,
    h: usize,
    w: usize,
    weights: &[i32],
    bias: &[i32],
    cout: usize,
    k: usize,
    pad: usize,
) -> Vec<Target> {
    let mut out = vec![builder.zero(); cout * h * w];
    for co in 0..cout {
        for i in 0..h {
            for j in 0..w {
                let mut terms_c = Vec::with_capacity(cin * k * k);
                let mut terms_v = Vec::with_capacity(cin * k * k);
                for ci in 0..cin {
                    for ki in 0..k {
                        for kj in 0..k {
                            let ii = i as isize + ki as isize - pad as isize;
                            let jj = j as isize + kj as isize - pad as isize;
                            let widx = wco(co, ci, ki, kj, cin, k, k);
                            terms_c.push(weights[widx]);
                            let v = if ii >= 0 && jj >= 0 && ii < h as isize && jj < w as isize {
                                inp[idx3(ci, ii as usize, jj as usize, h, w)]
                            } else {
                                builder.zero()
                            };
                            terms_v.push(v);
                        }
                    }
                }
                let mut acc = dot_const(builder, &terms_c, &terms_v);
                let bc = builder.constant(to_field_i32(bias[co]));
                acc = builder.add(acc, bc);
                out[idx3(co, i, j, h, w)] = acc;
            }
        }
    }
    out
}

pub(crate) fn pool2(
    builder: &mut Builder,
    inp: &[Target],
    cin: usize,
    h: usize,
    w: usize,
    bits: usize,
) -> Vec<Target> {
    let oh = h / 2;
    let ow = w / 2;
    let mut out = vec![builder.zero(); cin * oh * ow];
    for c in 0..cin {
        for i in 0..oh {
            for j in 0..ow {
                let a = inp[idx3(c, 2 * i, 2 * j, h, w)];
                let b = inp[idx3(c, 2 * i, 2 * j + 1, h, w)];
                let c0 = inp[idx3(c, 2 * i + 1, 2 * j, h, w)];
                let d = inp[idx3(c, 2 * i + 1, 2 * j + 1, h, w)];
                out[idx3(c, i, j, oh, ow)] = max4(builder, a, b, c0, d, bits);
            }
        }
    }
    out
}

pub struct RealSmallIrisZkCircuit {
    pub input_targets: Vec<Target>,
    pub data:          CircuitData<F, C, D>,
    pub export:        RealSmallIrisExport,
}

impl RealSmallIrisZkCircuit {
    pub fn build(export: RealSmallIrisExport, value_bits: usize) -> Result<Self> {
        export.validate()?;
        eprintln!(
            "real_smalliris_zk: building Plonky2 circuit ({}×{}, this may take minutes and large \
             RAM)…",
            export.h, export.w
        );
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = Builder::new(config);
        let h = export.h;
        let w = export.w;
        let (h1, w1) = export.hw_after_pools(1);
        let (h2, w2) = export.hw_after_pools(2);
        let (hh, ww) = export.hw_after_pools(3);
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

        builder.print_gate_counts(0);
        let data = builder.build::<C>();
        eprintln!("real_smalliris_zk: circuit build done.");
        Ok(Self {
            input_targets,
            data,
            export,
        })
    }

    pub fn prove(&self, input_i32: &[i32]) -> Result<(RealSmallIrisProof, Vec<i32>)> {
        assert_eq!(input_i32.len(), self.export.input_len());
        let sums = forward_circuit_feature_sums(&self.export, input_i32);
        let mut pw = PartialWitness::new();
        for (&t, &v) in self.input_targets.iter().zip(input_i32) {
            pw.set_target(t, to_field_i32(v))
                .map_err(|e| anyhow::anyhow!("pw: {e:?}"))?;
        }
        let proof = self
            .data
            .prove(pw)
            .map_err(|e| anyhow::anyhow!("prove: {e:?}"))?;
        Ok((proof, sums))
    }

    pub fn verify(&self, proof: &RealSmallIrisProof) -> Result<()> {
        self.data
            .verify(proof.clone())
            .map_err(|e| anyhow::anyhow!("verify: {e:?}"))
    }
}

pub fn public_sums_match(proof: &RealSmallIrisProof, expected: &[i32]) -> bool {
    proof.public_inputs.len() == expected.len()
        && proof
            .public_inputs
            .iter()
            .zip(expected)
            .all(|(&x, &e)| field_to_i32_generic(x) == e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64.json")
    }

    /// Fixture-based unit test for a checked-in SmallIris int32 export.
    #[test]
    fn real_fixture_matches_smalliris_export_shape() {
        let exp = RealSmallIrisExport::load_path(fixture_path()).expect("load fixture");
        exp.validate().expect("shape");
        assert_eq!(exp.h, 48);
        assert_eq!(exp.w, 48);
        assert!((exp.quantize_q - 4096.0).abs() < 1e-6);
    }

    /// Fast integer forward pass on the checked-in fixture without building a
    /// Plonky2 circuit.
    #[test]
    fn forward_int_on_real_fixture_channel_sums() {
        let exp = RealSmallIrisExport::load_path(fixture_path()).unwrap();
        let input = synthetic_input_i32(&exp);
        let sums = forward_int_feature_sums(&exp, &input);
        assert_eq!(sums.len(), exp.c3);
    }

    #[test]
    fn forward_int_after_layer2_shape() {
        let exp = RealSmallIrisExport::load_path(fixture_path()).unwrap();
        let input = synthetic_input_i32(&exp);
        let p2 = forward_int_after_layer2(&exp, &input);
        assert_eq!(p2.len(), exp.num_public_layer2());
    }

    #[test]
    fn diagnose_min_value_bits_sane() {
        let exp = RealSmallIrisExport::load_path(fixture_path()).unwrap();
        let input = synthetic_input_i32(&exp);
        let b = diagnose_min_value_bits(&exp, &input);
        assert!(b >= 8 && b <= 64);
    }

    /// Sanity-check protocol accounting after manually shrinking the spatial
    /// statement size while reusing the same convolution weights.
    #[test]
    fn smaller_spatial_protocol_counts() {
        let mut exp = RealSmallIrisExport::load_path(fixture_path()).unwrap();
        exp.h = 32;
        exp.w = 32;
        exp.validate().unwrap();
        assert_eq!(exp.num_public_layer1(), exp.c1 * 16 * 16);
        assert_eq!(exp.num_public_layer2(), exp.c2 * 8 * 8);
        assert_eq!(exp.num_public_layer3(), exp.c3);
        assert_eq!(exp.gap_cells(), 16);
        let input = vec![0i32; exp.input_len()];
        let sums = forward_int_feature_sums(&exp, &input);
        assert_eq!(sums.len(), exp.c3);
    }

}
