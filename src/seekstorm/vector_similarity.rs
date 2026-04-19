#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt;

use crate::vector::{Embedding, EmbeddingView};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Similarity measure for comparing vector embeddings.
#[derive(Clone, Copy, Default, Debug, PartialEq, Deserialize, Serialize, ToSchema)]
pub enum VectorSimilarity {
    /// Cosine Similarity
    /// Cosine similarity measures only the direction (angle) between vectors.
    /// At index time the vectors are normalized to a norm of 1.0, so that the cosine similarity can be computed efficiently as a dot product without needing to compute the norms at query time.
    /// The query vector is normalized to a norm of 1.0, so that the cosine similarity can be computed efficiently as a dot product without needing to compute the norms at query time.
    Cosine,
    /// Dot product (Inner Product),  
    /// Dot product measures both the direction and the magnitude (length) of vectors.
    /// Use dot product, when using pre-normalized external embeddings, to prevent unnecessary normalization with cosine similarity.
    /// Dot product and cosine similarity produce the same ranking for L2-normalized vectors.
    #[default]
    Dot,
    /// Euclidean distance (L2)
    Euclidean,
}

impl fmt::Display for VectorSimilarity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VectorSimilarity::Cosine => write!(f, "Cosine"),
            VectorSimilarity::Dot => write!(f, "Dot"),
            VectorSimilarity::Euclidean => write!(f, "Euclidean"),
        }
    }
}

///Specifies in which cluster to search for ANN results.
#[derive(Default, PartialEq, Clone, Debug, Serialize, Deserialize, ToSchema)]
pub enum AnnMode {
    /// Search in all clusters (default)
    #[default]
    All,
    /// Search only in the clusters with the highest similarity scores to the query vector.
    /// The number of clusters to search is specified by the n-probe parameter.
    /// You cannot directly set a specific, guaranteed recall number (e.g., "always give me 95% recall@10"). There is no one-fits-all, there is no automatism.
    /// Instead, you manually tune parameters that control the tradeoff between query latency and accuracy.
    /// Because recall depends heavily on the structure of your specific data (distribution, dimensionality, and clustering) and queries,
    /// there is always a trial-and-error (benchmarking) phase required to determine the right settings for your data.
    /// Examples:
    /// wikipedia, VectorSimilarity::Dot, dimensions: 64, Precision::F32, Clustering::Auto, Clustering::I8, recall@10=95% -> Nprobe(55)
    /// wikipedia, VectorSimilarity::Dot, dimensions: 64, Precision::F32, Clustering::Auto, Clustering::I8, recall@10=99% -> Nprobe(140)
    /// sift1m, VectorSimilarity::Euclidean, dimensions: 128, Precision::F32, Clustering::Auto, Quantization::None, recall@10=95% -> Nprobe(11)
    /// sift1m, VectorSimilarity::Euclidean, dimensions: 128, Precision::F32, Clustering::Auto, Quantization::None, recall@10=99% -> Nprobe(22)
    Nprobe(usize),
    /// Search only in clusters with similarity scores to the query vector above the specified threshold.
    /// For dot product similarity, the similarity threshold should be between 0.0 and 1.0, where higher values indicate higher similarity (identical=1.0).
    /// For Euclidean distance similarity, the similarity threshold should be between 0.0 and infinity, where lower values indicate higher similarity (identical=0.0).
    Similaritythreshold(f32),
    /// Search only in the clusters with the highest similarity scores to the query vector, but only if their similarity scores are above the specified threshold, and up to the number of clusters specified by the n-probe parameter.
    /// For dot product similarity, the similarity threshold should be between 0.0 and 1.0, where higher values indicate higher similarity (identical=1.0).
    /// For Euclidean distance similarity, the similarity threshold should be between 0.0 and infinity, where lower values indicate higher similarity (identical=0.0).
    NprobeSimilaritythreshold(usize, f32),
}

pub(crate) fn normalize_f32(embeddings: &mut [f32]) {
    let norm = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_factor = 1.0 / norm;
    embeddings.iter_mut().for_each(|b| *b *= norm_factor);
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn normalize_f32_avx2(data: &mut [f32]) {
    unsafe {
        let len = data.len();
        if len == 0 {
            return;
        }

        let mut sum_sq;
        let chunks = data.chunks_exact(8);
        let rem = chunks.remainder();

        let mut sum_vec = _mm256_setzero_ps();
        for chunk in chunks {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v, v));
        }

        sum_sq = horizontal_sum_avx2(sum_vec);
        for &x in rem {
            sum_sq += x * x;
        }

        let inv_norm = 1.0 / sum_sq.sqrt();
        let inv_norm_vec = _mm256_set1_ps(inv_norm);

        let chunks_mut = data.chunks_exact_mut(8);
        for chunk in chunks_mut {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            let res = _mm256_mul_ps(v, inv_norm_vec);
            _mm256_storeu_ps(chunk.as_mut_ptr(), res);
        }

        let rem_mut = data.chunks_exact_mut(8).into_remainder();
        for x in rem_mut {
            *x *= inv_norm;
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[allow(clippy::type_complexity)]
pub(crate) fn similarity_embedding_view(
    a: &Embedding,
    b: &EmbeddingView,
    scale_norm: Option<(f32, i32, i32, i32, f32, i32, i32, i32)>,
    vector_similarity: VectorSimilarity,
) -> f32 {
    match (a, vector_similarity) {
        (Embedding::I8(a), VectorSimilarity::Dot) => {
            if let EmbeddingView::I8(b) = b {
                if let Some((
                    query_scale,
                    _query_norm,
                    _query_zero_point,
                    _query_sum_q,
                    embedding_scale,
                    _embedding_norm,
                    _embedding_zero_point,
                    _embedding_sum_q,
                )) = scale_norm
                {
                    dot_i8_quantized(a, query_scale, b, embedding_scale)
                } else {
                    dot_i8(a, b) as f32
                }
            } else {
                panic!("dot_i8 only supports i8 embeddings")
            }
        }

        (Embedding::F32(a), VectorSimilarity::Dot) => {
            if let EmbeddingView::F32(b) = b {
                dot_f32(a, b)
            } else {
                panic!("dot_f32 only supports f32 embeddings")
            }
        }
        (Embedding::I8(a), VectorSimilarity::Cosine) => {
            if let EmbeddingView::I8(b) = b {
                dot_i8(a, b) as f32
            } else {
                panic!("dot_i8 only supports i8 embeddings1")
            }
        }
        (Embedding::F32(a), VectorSimilarity::Cosine) => {
            if let EmbeddingView::F32(b) = b {
                dot_f32(a, b)
            } else {
                panic!("dot_f32 only supports f32 embeddings")
            }
        }
        (Embedding::I8(a), VectorSimilarity::Euclidean) => {
            if let EmbeddingView::I8(b) = b {
                if let Some((
                    query_scale,
                    query_norm,
                    query_zero_point,
                    query_sum_q,
                    embedding_scale,
                    embedding_norm,
                    embedding_zero_point,
                    embedding_sum_q,
                )) = scale_norm
                {
                    -euclidean_i8_quantized_affine(
                        a,
                        query_scale,
                        query_norm,
                        query_zero_point,
                        query_sum_q,
                        b,
                        embedding_scale,
                        embedding_norm,
                        embedding_zero_point,
                        embedding_sum_q,
                    )
                } else {
                    -euclidean_i8(a, b)
                }
            } else {
                panic!("euclidean_i8 only supports i8 embeddings")
            }
        }
        (Embedding::F32(a), VectorSimilarity::Euclidean) => {
            if let EmbeddingView::F32(b) = b {
                -euclidean_f32(a, b)
            } else {
                panic!("euclidean_f32 only supports f32 embeddings")
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn similarity_embedding(
    a: &Embedding,
    b: &Embedding,
    scale_norm: Option<(f32, i32, i32, i32, f32, i32, i32, i32)>,
    vector_similarity: VectorSimilarity,
) -> f32 {
    match (a, vector_similarity) {
        (Embedding::I8(a), VectorSimilarity::Dot) => {
            if let Embedding::I8(b) = b {
                if let Some((
                    query_scale,
                    _query_norm,
                    _query_zero_point,
                    _query_sum_q,
                    embedding_scale,
                    _embedding_norm,
                    _embedding_zero_point,
                    _embedding_sum_q,
                )) = scale_norm
                {
                    dot_i8_quantized(a, query_scale, b, embedding_scale)
                } else {
                    dot_i8(a, b) as f32
                }
            } else {
                panic!("dot_i8 only supports i8 embeddings")
            }
        }

        (Embedding::F32(a), VectorSimilarity::Dot) => {
            if let Embedding::F32(b) = b {
                dot_f32(a, b)
            } else {
                panic!("dot_f32 only supports f32 embeddings")
            }
        }
        (Embedding::I8(a), VectorSimilarity::Cosine) => {
            if let Embedding::I8(b) = b {
                dot_i8(a, b) as f32
            } else {
                panic!("dot_i8 only supports i8 embeddings2")
            }
        }
        (Embedding::F32(a), VectorSimilarity::Cosine) => {
            if let Embedding::F32(b) = b {
                dot_f32(a, b)
            } else {
                panic!("dot_f32 only supports f32 embeddings")
            }
        }
        (Embedding::I8(a), VectorSimilarity::Euclidean) => {
            if let Embedding::I8(b) = b {
                if let Some((
                    query_scale,
                    query_norm,
                    query_zero_point,
                    query_sum_q,
                    embedding_scale,
                    embedding_norm,
                    embedding_zero_point,
                    embedding_sum_q,
                )) = scale_norm
                {
                    -euclidean_i8_quantized_affine(
                        a,
                        query_scale,
                        query_norm,
                        query_zero_point,
                        query_sum_q,
                        b,
                        embedding_scale,
                        embedding_norm,
                        embedding_zero_point,
                        embedding_sum_q,
                    )
                } else {
                    -euclidean_i8(a, b)
                }
            } else {
                panic!("euclidean_i8 only supports i8 embeddings")
            }
        }
        (Embedding::F32(a), VectorSimilarity::Euclidean) => {
            if let Embedding::F32(b) = b {
                -euclidean_f32(a, b)
            } else {
                panic!("euclidean_f32 only supports f32 embeddings")
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) unsafe fn similarity_avx2_embedding_view(
    query: &QuerySimd,
    emb: &EmbeddingView,
    scale_norm: Option<(f32, i32, i32, i32, f32, i32, i32, i32)>,
    vector_similarity: VectorSimilarity,
) -> f32 {
    unsafe {
        match (emb, vector_similarity) {
            (EmbeddingView::I8(e), VectorSimilarity::Dot) => {
                if let Some((
                    query_scale,
                    _query_norm,
                    _query_zero_point,
                    _query_sum_q,
                    embedding_scale,
                    _embedding_norm,
                    _embedding_zero_point,
                    _embedding_sum_q,
                )) = scale_norm
                {
                    dot_i8_quantized_avx2(query, query_scale, e, embedding_scale)
                } else {
                    dot_i8_avx2(query, e) as f32
                }
            }

            (EmbeddingView::F32(e), VectorSimilarity::Dot) => dot_f32_avx2(query, e),
            (EmbeddingView::I8(e), VectorSimilarity::Cosine) => dot_i8_avx2(query, e) as f32,
            (EmbeddingView::F32(e), VectorSimilarity::Cosine) => dot_f32_avx2(query, e),
            (EmbeddingView::I8(e), VectorSimilarity::Euclidean) => {
                if let Some((
                    query_scale,
                    query_norm,
                    query_zero_point,
                    query_sum_q,
                    embedding_scale,
                    embedding_norm,
                    embedding_zero_point,
                    embedding_sum_q,
                )) = scale_norm
                {
                    -euclidean_i8_quantized_affine_avx2(
                        query,
                        query_scale,
                        query_norm,
                        query_zero_point,
                        query_sum_q,
                        e,
                        embedding_scale,
                        embedding_norm,
                        embedding_zero_point,
                        embedding_sum_q,
                    )
                } else {
                    -euclidean_i8_avx2(query, e) as f32
                }
            }
            (EmbeddingView::F32(e), VectorSimilarity::Euclidean) => -euclidean_f32_avx2(query, e),
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) unsafe fn similarity_avx2_embedding(
    query: &QuerySimd,
    emb: &Embedding,
    scale_norm: Option<(f32, i32, i32, i32, f32, i32, i32, i32)>,
    vector_similarity: VectorSimilarity,
) -> f32 {
    unsafe {
        match (emb, vector_similarity) {
            (Embedding::I8(e), VectorSimilarity::Dot) => {
                if let Some((
                    query_scale,
                    _query_norm,
                    _query_zero_point,
                    _query_sum_q,
                    embedding_scale,
                    _embedding_norm,
                    _embedding_zero_point,
                    _embedding_sum_q,
                )) = scale_norm
                {
                    dot_i8_quantized_avx2(query, query_scale, e, embedding_scale)
                } else {
                    dot_i8_avx2(query, e) as f32
                }
            }

            (Embedding::F32(e), VectorSimilarity::Dot) => dot_f32_avx2(query, e),
            (Embedding::I8(e), VectorSimilarity::Cosine) => dot_i8_avx2(query, e) as f32,
            (Embedding::F32(e), VectorSimilarity::Cosine) => dot_f32_avx2(query, e),
            (Embedding::I8(e), VectorSimilarity::Euclidean) => {
                if let Some((
                    query_scale,
                    query_norm,
                    query_zero_point,
                    query_sum_q,
                    embedding_scale,
                    embedding_norm,
                    embedding_zero_point,
                    embedding_sum_q,
                )) = scale_norm
                {
                    -euclidean_i8_quantized_affine_avx2(
                        query,
                        query_scale,
                        query_norm,
                        query_zero_point,
                        query_sum_q,
                        e,
                        embedding_scale,
                        embedding_norm,
                        embedding_zero_point,
                        embedding_sum_q,
                    )
                } else {
                    -euclidean_i8_avx2(query, e) as f32
                }
            }
            (Embedding::F32(e), VectorSimilarity::Euclidean) => -euclidean_f32_avx2(query, e),
        }
    }
}

pub(crate) fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vektoren müssen die gleiche Länge haben");

    let sum_squared_diffs: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();

    sum_squared_diffs
}

pub(crate) fn euclidean_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vektoren müssen die gleiche Länge haben");

    let sum_squared_diffs: i32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).pow(2))
        .sum();

    sum_squared_diffs as f32
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn euclidean_f32_avx2(query: &QuerySimd, b: &[f32]) -> f32 {
    unsafe {
        let query = match query {
            QuerySimd::F(e) => e,
            _ => {
                println!("{:?}", query);
                panic!("euclidean_avx2_f32 only supports f32 embeddings")
            }
        };

        let mut sum_v = _mm256_setzero_ps();

        for (i, q) in query.iter().enumerate() {
            let va = *q;
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));

            let diff = _mm256_sub_ps(va, vb);
            let squared = _mm256_mul_ps(diff, diff);
            sum_v = _mm256_add_ps(sum_v, squared);
        }

        let mut chunks = [0.0f32; 8];
        _mm256_storeu_ps(chunks.as_mut_ptr(), sum_v);
        let total_sum: f32 = chunks.iter().sum();

        total_sum
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn euclidean_i8_avx2(query: &QuerySimd, b: &[i8]) -> i32 {
    unsafe {
        let query = match query {
            QuerySimd::I(e) => e,
            _ => {
                println!("{:?}", query);
                panic!("euclidean_avx2_i8 only supports i8 embeddings")
            }
        };

        let mut sum_v = _mm256_setzero_si256();

        for (i, q) in query.iter().enumerate() {
            let va = *q;
            let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i);

            let va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
            let vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
            let diff_lo = _mm256_sub_epi16(va_lo, vb_lo);

            let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            let diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

            let sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
            let sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

            sum_v = _mm256_add_epi32(sum_v, sq_lo);
            sum_v = _mm256_add_epi32(sum_v, sq_hi);
        }

        let mut chunks = [0i32; 8];
        _mm256_storeu_si256(chunks.as_mut_ptr() as *mut __m256i, sum_v);
        let total_sum: i32 = chunks.iter().sum();

        total_sum
    }
}

pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

#[derive(Clone, Debug)]
pub(crate) enum QuerySimd {
    I(Vec<__m256i>),
    F(Vec<__m256>),
}

impl QuerySimd {
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn new(query: &Embedding) -> Self {
        unsafe {
            match query {
                Embedding::I8(e) => {
                    let mut q = Vec::with_capacity(e.len() >> 5);
                    for i in (0..e.len()).step_by(32) {
                        q.push(_mm256_loadu_si256(e.as_ptr().add(i) as *const __m256i));
                    }

                    QuerySimd::I(q)
                }
                Embedding::F32(e) => {
                    let mut q = Vec::with_capacity(e.len() >> 3);
                    for i in (0..e.len()).step_by(8) {
                        q.push(_mm256_loadu_ps(e.as_ptr().add(i)));
                    }

                    QuerySimd::F(q)
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn dot_f32_avx2(query: &QuerySimd, emb: &[f32]) -> f32 {
    unsafe {
        let query = match query {
            QuerySimd::F(e) => e,
            _ => {
                println!("{:?}", query);
                panic!("dot_f32 only supports f32 embeddings")
            }
        };

        let mut sum = _mm256_setzero_ps();

        for (i, q) in query.iter().enumerate() {
            let va = *q;
            let vb = _mm256_loadu_ps(emb.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
        tmp.iter().sum()
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_i8_avx2(query: &QuerySimd, emb: &[i8]) -> i32 {
    unsafe {
        let query = match query {
            QuerySimd::I(e) => e,
            _ => {
                println!("{:?}", query);
                panic!("dot_i8 only supports i8 embeddings3")
            }
        };

        let mut acc = _mm256_setzero_si256();
        for (i, q) in query.iter().enumerate() {
            let v0 = _mm256_loadu_si256(emb.as_ptr().add(i * 32) as *const __m256i);

            let q0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(*q));
            let v0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v0));

            let q0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(*q, 1));
            let v0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v0, 1));

            let prod0_lo = _mm256_madd_epi16(q0_lo, v0_lo);
            let prod0_hi = _mm256_madd_epi16(q0_hi, v0_hi);

            acc = _mm256_add_epi32(acc, prod0_lo);
            acc = _mm256_add_epi32(acc, prod0_hi);
        }

        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);

        tmp.iter().sum()
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn quantize_avx2_f32_to_i8(input: &[f32]) -> Embedding {
    unsafe {
        let mut output = vec![0i8; input.len()];
        assert_eq!(input.len(), output.len());

        let n = input.len();
        let mut i = 0;

        let scale_v = _mm256_set1_ps(127.0);

        while i + 32 <= n {
            let x0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let x1 = _mm256_loadu_ps(input.as_ptr().add(i + 8));
            let x2 = _mm256_loadu_ps(input.as_ptr().add(i + 16));
            let x3 = _mm256_loadu_ps(input.as_ptr().add(i + 24));

            let x0 = _mm256_mul_ps(x0, scale_v);
            let x1 = _mm256_mul_ps(x1, scale_v);
            let x2 = _mm256_mul_ps(x2, scale_v);
            let x3 = _mm256_mul_ps(x3, scale_v);

            let i0 = _mm256_cvtps_epi32(x0);
            let i1 = _mm256_cvtps_epi32(x1);
            let i2 = _mm256_cvtps_epi32(x2);
            let i3 = _mm256_cvtps_epi32(x3);

            let i01 = _mm256_packs_epi32(i0, i1);
            let i23 = _mm256_packs_epi32(i2, i3);

            let packed = _mm256_packs_epi16(i01, i23);

            _mm256_storeu_si256(output.as_mut_ptr().add(i) as *mut __m256i, packed);

            i += 32;
        }

        for j in i..n {
            output[j] = (input[j] * 127.0).round().clamp(-127.0, 127.0) as i8;
        }
        Embedding::I8(output)
    }
}

pub(crate) fn quantize_f32_to_i8(embedding: &[f32]) -> Embedding {
    let output = embedding
        .iter()
        .map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Embedding::I8(output)
}

#[derive(Debug, Clone)]
pub(crate) struct QuantizedVector {
    pub(crate) data: Vec<i8>,
    pub(crate) scale: f32,
    pub(crate) norm: i32,
    pub(crate) zero_point: i32,
    pub(crate) sum_q: i32,
}

impl QuantizedVector {
    pub(crate) fn new_scale(values: &[f32]) -> Self {
        let max_val = values.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let scale = max_val / 127.0;

        let data: Vec<i8> = values.iter().map(|&x| (x / scale).round() as i8).collect();

        Self {
            data,
            scale,
            norm: 0,
            zero_point: 0,
            sum_q: 0,
        }
    }

    pub(crate) fn new_scale_avx2(values: &[f32]) -> Self {
        unsafe {
            let max_val = Self::max_abs_avx2(values);
            let scale = max_val / 127.0;

            let data = Self::quantize_avx2(values, scale);

            Self {
                data,
                scale,
                norm: 0,
                zero_point: 0,
                sum_q: 0,
            }
        }
    }

    pub(crate) fn new_scale_norm_affine(
        min_vector_value: &mut f32,
        max_vector_value: &mut f32,
        values: &[f32],
    ) -> Self {
        let (mut min_val, mut max_val) = values
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min_v, max_v), x| {
                (min_v.min(x), max_v.max(x))
            });

        if min_val < *min_vector_value {
            *min_vector_value = min_val;
        } else {
            min_val = *min_vector_value;
        }

        if max_val > *max_vector_value {
            let max_val_power_of_two =
                ((((max_val - min_val) as i64) as u64).next_power_of_two() - 1) as f32;
            *max_vector_value = max_val_power_of_two;
        } else {
            max_val = *max_vector_value;
        }

        let range = ((((max_val - min_val) as i64) as u64).next_power_of_two() - 1) as f32;
        let scale = range / 255.0;
        let zero_point_f = -128.0 - (min_val / scale);
        let zero_point = zero_point_f.round().clamp(-128.0, 127.0) as i32;

        let data: Vec<i8> = values
            .iter()
            .map(|&x| ((x / scale).round() as i32 + zero_point).clamp(-128, 127) as i8)
            .collect();

        let squared_norm = data.iter().map(|&x| x as i32 * x as i32).sum();
        let sum_q = data.iter().map(|x| *x as i32).sum::<i32>();

        Self {
            data,
            scale,
            norm: squared_norm,
            zero_point,
            sum_q,
        }
    }

    pub(crate) fn new_scale_norm_affine_avx2(
        min_vector_value: &mut f32,
        max_vector_value: &mut f32,
        values: &[f32],
    ) -> Self {
        unsafe {
            let mut max_val = Self::max_avx2(values);
            let mut min_val = Self::min_avx2(values);

            if min_val < *min_vector_value {
                *min_vector_value = min_val;
            } else {
                min_val = *min_vector_value;
            }

            if max_val > *max_vector_value {
                let max_val_power_of_two =
                    ((((max_val - min_val) as i64) as u64).next_power_of_two() - 1) as f32;
                *max_vector_value = max_val_power_of_two;
            } else {
                max_val = *max_vector_value;
            }

            let range = ((((max_val - min_val) as i64) as u64).next_power_of_two() - 1) as f32;
            let scale = range / 255.0;
            let zero_point_f = -128.0 - (min_val / scale);
            let zero_point = zero_point_f.round().clamp(-128.0, 127.0) as i32;

            let data = Self::quantize_affine_avx2(values, scale, zero_point);

            let squared_norm = Self::squared_norm_avx2(&data);
            let sum_q = Self::sum_avx2(&data);

            Self {
                data,
                scale,
                norm: squared_norm,
                zero_point,
                sum_q,
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_abs_avx2(values: &[f32]) -> f32 {
        unsafe {
            let mut max_vec = _mm256_setzero_ps();
            let mut i = 0;

            while i + 8 <= values.len() {
                let v = _mm256_loadu_ps(values.as_ptr().add(i));

                let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);

                max_vec = _mm256_max_ps(max_vec, abs);
                i += 8;
            }

            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);

            let mut max_val = tmp.iter().copied().fold(0.0, f32::max);

            for &x in &values[i..] {
                max_val = max_val.max(x.abs());
            }

            max_val
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_avx2(values: &[f32]) -> f32 {
        unsafe {
            if values.is_empty() {
                return f32::NEG_INFINITY;
            }

            let mut i = 0;

            let mut max_vec = if values.len() >= 8 {
                let v = _mm256_loadu_ps(values.as_ptr());
                i = 8;
                v
            } else {
                _mm256_set1_ps(f32::NEG_INFINITY)
            };

            while i + 8 <= values.len() {
                let v = _mm256_loadu_ps(values.as_ptr().add(i));
                max_vec = _mm256_max_ps(max_vec, v);
                i += 8;
            }

            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);

            let mut max_val = tmp.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            for &x in &values[i..] {
                max_val = max_val.max(x);
            }

            max_val
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn min_avx2(values: &[f32]) -> f32 {
        unsafe {
            if values.is_empty() {
                return f32::INFINITY;
            }

            let mut i = 0;

            let mut min_vec = if values.len() >= 8 {
                let v = _mm256_loadu_ps(values.as_ptr());
                i = 8;
                v
            } else {
                _mm256_set1_ps(f32::INFINITY)
            };

            while i + 8 <= values.len() {
                let v = _mm256_loadu_ps(values.as_ptr().add(i));
                min_vec = _mm256_min_ps(min_vec, v);
                i += 8;
            }

            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), min_vec);

            let mut min_val = tmp.iter().copied().fold(f32::INFINITY, f32::min);

            for &x in &values[i..] {
                min_val = min_val.min(x);
            }

            min_val
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2(values: &[f32], scale: f32) -> Vec<i8> {
        unsafe {
            let inv_scale = 1.0 / scale;
            let scale_vec = _mm256_set1_ps(inv_scale);

            let mut result = vec![0i8; values.len()];
            let mut i = 0;

            while i + 16 <= values.len() {
                let v0 = _mm256_loadu_ps(values.as_ptr().add(i));
                let v1 = _mm256_loadu_ps(values.as_ptr().add(i + 8));

                let s0 = _mm256_mul_ps(v0, scale_vec);
                let s1 = _mm256_mul_ps(v1, scale_vec);

                let i0 = _mm256_cvtps_epi32(s0);
                let i1 = _mm256_cvtps_epi32(s1);

                let p16 = _mm256_packs_epi32(i0, i1);

                let p8 = _mm256_packs_epi16(p16, p16);

                let lo = _mm256_extracti128_si256(p8, 0);
                let hi = _mm256_extracti128_si256(p8, 1);

                _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, lo);
                _mm_storeu_si128(result.as_mut_ptr().add(i + 8) as *mut __m128i, hi);

                i += 16;
            }

            for j in i..values.len() {
                result[j] = (values[j] * inv_scale).round() as i8;
            }

            result
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn round_away_from_zero_avx2(x: __m256) -> __m256i {
        let half = _mm256_set1_ps(0.5);

        let sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0));

        let signed_half = _mm256_or_ps(half, sign);

        let adjusted = _mm256_add_ps(x, signed_half);

        _mm256_cvttps_epi32(adjusted)
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn quantize_affine_avx2(
        values: &[f32],
        scale: f32,
        zero_point: i32,
    ) -> Vec<i8> {
        unsafe {
            let inv_scale = 1.0 / scale;

            let scale_vec = _mm256_set1_ps(inv_scale);
            let zp_vec = _mm256_set1_epi32(zero_point);

            let min_i32 = _mm256_set1_epi32(-128);
            let max_i32 = _mm256_set1_epi32(127);

            let mut result = vec![0i8; values.len()];
            let mut i = 0;

            while i + 16 <= values.len() {
                let v0 = _mm256_loadu_ps(values.as_ptr().add(i));
                let v1 = _mm256_loadu_ps(values.as_ptr().add(i + 8));

                let s0 = _mm256_mul_ps(v0, scale_vec);
                let s1 = _mm256_mul_ps(v1, scale_vec);

                let mut i0 = Self::round_away_from_zero_avx2(s0);
                let mut i1 = Self::round_away_from_zero_avx2(s1);

                i0 = _mm256_add_epi32(i0, zp_vec);
                i1 = _mm256_add_epi32(i1, zp_vec);

                i0 = _mm256_max_epi32(i0, min_i32);
                i0 = _mm256_min_epi32(i0, max_i32);
                i1 = _mm256_max_epi32(i1, min_i32);
                i1 = _mm256_min_epi32(i1, max_i32);

                let i0_lo = _mm256_castsi256_si128(i0);
                let i0_hi = _mm256_extracti128_si256(i0, 1);
                let i1_lo = _mm256_castsi256_si128(i1);
                let i1_hi = _mm256_extracti128_si256(i1, 1);

                let p0 = _mm_packs_epi32(i0_lo, i0_hi);
                let p1 = _mm_packs_epi32(i1_lo, i1_hi);

                let p = _mm_packs_epi16(p0, p1);

                _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, p);

                i += 16;
            }

            for j in i..values.len() {
                let q = ((values[j] * inv_scale).round() as i32 + zero_point).clamp(-128, 127);
                result[j] = q as i8;
            }

            result
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn squared_norm_avx2(data: &[i8]) -> i32 {
        unsafe {
            let mut sum = _mm256_setzero_si256();
            let mut i = 0;

            while i + 32 <= data.len() {
                let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

                let v_lo = _mm256_castsi256_si128(v);
                let v_hi = _mm256_extracti128_si256(v, 1);

                let lo = _mm256_cvtepi8_epi16(v_lo);
                let hi = _mm256_cvtepi8_epi16(v_hi);

                let lo_sq = _mm256_madd_epi16(lo, lo);
                let hi_sq = _mm256_madd_epi16(hi, hi);

                sum = _mm256_add_epi32(sum, lo_sq);
                sum = _mm256_add_epi32(sum, hi_sq);

                i += 32;
            }

            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum);

            let mut total: i32 = tmp.iter().sum();

            for &x in &data[i..] {
                total += (x as i32) * (x as i32);
            }

            total
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sum_avx2(data: &[i8]) -> i32 {
        unsafe {
            let mut sum = _mm256_setzero_si256();
            let mut i = 0;

            while i + 32 <= data.len() {
                let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

                let lo128 = _mm256_castsi256_si128(v);
                let hi128 = _mm256_extracti128_si256(v, 1);

                let lo = _mm256_cvtepi8_epi16(lo128);
                let hi = _mm256_cvtepi8_epi16(hi128);

                let ones = _mm256_set1_epi16(1);
                let lo_sum = _mm256_madd_epi16(lo, ones);
                let hi_sum = _mm256_madd_epi16(hi, ones);

                sum = _mm256_add_epi32(sum, lo_sum);
                sum = _mm256_add_epi32(sum, hi_sum);

                i += 32;
            }

            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum);
            let mut total: i32 = tmp.iter().sum();

            for &x in &data[i..] {
                total += x as i32;
            }

            total
        }
    }
}

fn dot_i8_quantized(v1: &[i8], scale1: f32, v2: &[i8], scale2: f32) -> f32 {
    let dot_i32: i32 = v1.iter().zip(v2).map(|(&a, &b)| a as i32 * b as i32).sum();

    dot_i32 as f32 * scale1 * scale2
}

fn dot_i8_quantized_avx2(v1: &QuerySimd, scale1: f32, v2: &[i8], scale2: f32) -> f32 {
    let dot_i32: i32 = unsafe { dot_i8_avx2(v1, v2) };

    dot_i32 as f32 * scale1 * scale2
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn euclidean_i8_quantized_affine(
    v1: &[i8],
    scale1: f32,
    norm1: i32,
    zero_point1: i32,
    sum_q1: i32,
    v2: &[i8],
    scale2: f32,
    norm2: i32,
    zero_point2: i32,
    sum_q2: i32,
) -> f32 {
    let dot_i32: i32 = v1.iter().zip(v2).map(|(&a, &b)| a as i32 * b as i32).sum();

    let n = v1.len() as i32;

    let dot_i32 =
        dot_i32 - zero_point2 * sum_q1 - zero_point1 * sum_q2 + n * zero_point1 * zero_point2;

    let norm1 = norm1 - 2 * zero_point1 * sum_q1 + n * zero_point1 * zero_point1;

    let norm2 = norm2 - 2 * zero_point2 * sum_q2 + n * zero_point2 * zero_point2;

    let dot = dot_i32 as f32 * scale1 * scale2;

    let norm1 = norm1 as f32 * scale1 * scale1;
    let norm2 = norm2 as f32 * scale2 * scale2;
    (norm1 + norm2 - 2.0 * dot).max(0.0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn euclidean_i8_quantized_affine_avx2(
    v1: &QuerySimd,
    scale1: f32,
    norm1: i32,
    zero_point1: i32,
    sum_q1: i32,
    v2: &[i8],
    scale2: f32,
    norm2: i32,
    zero_point2: i32,
    sum_q2: i32,
) -> f32 {
    let dot_i32: i32 = unsafe { dot_i8_avx2(v1, v2) };

    let n = v2.len() as i32;

    let dot_i32 =
        dot_i32 - zero_point2 * sum_q1 - zero_point1 * sum_q2 + n * zero_point1 * zero_point2;

    let norm1 = norm1 - 2 * zero_point1 * sum_q1 + n * zero_point1 * zero_point1;

    let norm2 = norm2 - 2 * zero_point2 * sum_q2 + n * zero_point2 * zero_point2;

    let dot = dot_i32 as f32 * scale1 * scale2;

    let norm1 = norm1 as f32 * scale1 * scale1;
    let norm2 = norm2 as f32 * scale2 * scale2;
    (norm1 + norm2 - 2.0 * dot).max(0.0)
}
