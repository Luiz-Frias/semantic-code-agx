//! ONNX embedding adapter (local).

use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::TensorRef;
use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector, EmbeddingVectorFixed,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::Deserialize;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

const DEFAULT_TEST_TEXT: &str = "dimension probe";
const DEFAULT_MODEL_FILE: &str = "model.onnx";
const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";
const DEFAULT_INPUT_IDS: &str = "input_ids";
const DEFAULT_ATTENTION_MASK: &str = "attention_mask";

/// ONNX embedding adapter configuration.
#[derive(Debug, Clone)]
pub struct OnnxEmbeddingConfig {
    /// Directory containing `tokenizer.json` and ONNX model artifacts.
    pub model_dir: PathBuf,
    /// Optional model filename override.
    pub model_filename: Option<Box<str>>,
    /// Optional tokenizer filename override.
    pub tokenizer_filename: Option<Box<str>>,
    /// Optional expected embedding dimension.
    pub dimension: Option<u32>,
    /// Number of ONNX sessions to keep in the pool.
    pub session_pool_size: usize,
}

impl OnnxEmbeddingConfig {
    /// Create a config with the provided model directory.
    #[must_use]
    pub const fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            model_filename: None,
            tokenizer_filename: None,
            dimension: None,
            session_pool_size: 1,
        }
    }
}

/// ONNX embedding adapter implementation.
pub struct OnnxEmbedding {
    provider: EmbeddingProviderInfo,
    tokenizer: Arc<Tokenizer>,
    inputs: ModelInputs,
    output_name: Box<str>,
    session_pool: Arc<SessionPool>,
    dimension_override: Option<u32>,
    model_dimension: Option<u32>,
}

struct SessionPool {
    primary: Mutex<SessionSlot>,
    extras: Vec<Mutex<SessionSlot>>,
    next: AtomicUsize,
}

struct SessionSlot {
    session: Session,
    buffers: BatchBuffers,
}

impl SessionSlot {
    fn new(session: Session) -> Self {
        Self {
            session,
            buffers: BatchBuffers::default(),
        }
    }
}

impl SessionPool {
    fn new(primary: Session, extras: Vec<Session>) -> Self {
        Self {
            primary: Mutex::new(SessionSlot::new(primary)),
            extras: extras
                .into_iter()
                .map(SessionSlot::new)
                .map(Mutex::new)
                .collect(),
            next: AtomicUsize::new(0),
        }
    }

    fn pick(&self) -> &Mutex<SessionSlot> {
        if self.extras.is_empty() {
            return &self.primary;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        let slot = index % (self.extras.len() + 1);
        if slot == 0 {
            &self.primary
        } else {
            self.extras.get(slot - 1).unwrap_or(&self.primary)
        }
    }
}

impl OnnxEmbedding {
    /// Create a new ONNX embedding adapter.
    pub fn new(config: &OnnxEmbeddingConfig) -> Result<Self> {
        let model_dir = normalize_model_dir(&config.model_dir)?;
        let model_path = resolve_model_path(&model_dir, config.model_filename.as_deref())?;
        let tokenizer_path =
            resolve_tokenizer_path(&model_dir, config.tokenizer_filename.as_deref())?;

        let tokenizer = load_tokenizer(&tokenizer_path)?;
        let tokenizer_config = read_tokenizer_config(&model_dir)?;
        let model_config = read_model_config(&model_dir)?;
        let model_dimension = model_config
            .hidden_size
            .and_then(|value| u32::try_from(value).ok());

        let max_length = resolve_max_length(&tokenizer_config, &model_config);
        let tokenizer =
            configure_tokenizer(tokenizer, &tokenizer_config, &model_config, max_length)?;

        let pool_size = config.session_pool_size.max(1);
        let primary_session = Session::builder()
            .map_err(map_ort_error("onnx_session_builder_failed"))?
            .commit_from_file(&model_path)
            .map_err(map_ort_error("onnx_session_load_failed"))?;
        let inputs = resolve_inputs(primary_session.inputs())?;
        let output_name = primary_session
            .outputs()
            .first()
            .map(|outlet| outlet.name().to_owned().into_boxed_str())
            .ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "ONNX model outputs missing",
                    ErrorClass::NonRetriable,
                )
            })?;
        let mut extras = Vec::with_capacity(pool_size.saturating_sub(1));
        for _ in 1..pool_size {
            let session = Session::builder()
                .map_err(map_ort_error("onnx_session_builder_failed"))?
                .commit_from_file(&model_path)
                .map_err(map_ort_error("onnx_session_load_failed"))?;
            extras.push(session);
        }
        let session_pool = Arc::new(SessionPool::new(primary_session, extras));

        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("onnx").map_err(ErrorEnvelope::from)?,
            name: "ONNX".into(),
        };

        Ok(Self {
            provider,
            tokenizer: Arc::new(tokenizer),
            inputs,
            output_name,
            session_pool,
            dimension_override: config.dimension,
            model_dimension,
        })
    }

    fn embed_many_sync(
        tokenizer: &Tokenizer,
        inputs: &ModelInputs,
        output_name: &str,
        session_pool: &SessionPool,
        texts: &[Box<str>],
        expected_dimension: Option<u32>,
    ) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding input must be non-empty",
            ));
        }

        let mut text_inputs = Vec::with_capacity(texts.len());
        for text in texts {
            text_inputs.push(Cow::Borrowed(text.as_ref()));
        }
        let encodings = tokenizer.encode_batch(text_inputs, true).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                format!("tokenization failed: {error}"),
            )
        })?;
        let mut session_guard = session_pool.pick().lock().map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "ONNX session lock poisoned",
                ErrorClass::NonRetriable,
            )
        })?;
        let SessionSlot { session, buffers } = &mut *session_guard;
        let prepared = buffers.prepare_inputs(&encodings, inputs)?;
        let session_inputs = build_session_inputs(
            inputs,
            prepared.batch_size,
            prepared.sequence_length,
            prepared.input_ids,
            prepared.attention_mask,
            prepared.token_type_ids,
        )?;
        let outputs = session
            .run(session_inputs)
            .map_err(map_ort_error("onnx_inference_failed"))?;
        let output = outputs.get(output_name).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "ONNX model output missing",
                ErrorClass::NonRetriable,
            )
        })?;
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(map_ort_error("onnx_output_extract_failed"))?;

        let embeddings = match shape.len() {
            2 => map_pooled_embeddings(shape, data, prepared.batch_size, expected_dimension)?,
            3 => map_sequence_embeddings(
                shape,
                data,
                prepared.batch_size,
                prepared.sequence_length,
                prepared.attention_mask,
                expected_dimension,
            )?,
            _ => {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    format!("unexpected ONNX output rank {}", shape.len()),
                    ErrorClass::NonRetriable,
                ));
            },
        };
        drop(outputs);
        drop(session_guard);

        Ok(embeddings.into_iter().map(normalize_embedding).collect())
    }

    async fn run_blocking<T, F>(ctx: &RequestContext, operation: &'static str, work: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        ctx.ensure_not_cancelled(operation)?;
        let mut handle = tokio::task::spawn_blocking(work);
        tokio::select! {
            () = ctx.cancelled() => {
                handle.abort();
                Err(cancelled_error(operation))
            }
            result = &mut handle => {
                result.map_err(|error| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        format!("ONNX embed task failed: {error}"),
                        ErrorClass::NonRetriable,
                    )
                })?
            }
        }
    }
}

/// ONNX embedding adapter with a fixed compile-time dimension.
pub struct OnnxEmbeddingFixed<const D: usize> {
    inner: OnnxEmbedding,
}

impl<const D: usize> OnnxEmbeddingFixed<D> {
    /// Create a new ONNX adapter that enforces dimension `D`.
    pub fn new(config: &OnnxEmbeddingConfig) -> Result<Self> {
        let expected = Self::expected_dimension()?;
        let mut config = config.clone();
        if let Some(configured) = config.dimension {
            if configured != expected {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "embedding dimension mismatch",
                )
                .with_metadata("expected", expected.to_string())
                .with_metadata("actual", configured.to_string()));
            }
        } else {
            config.dimension = Some(expected);
        }

        Ok(Self {
            inner: OnnxEmbedding::new(&config)?,
        })
    }

    fn expected_dimension() -> Result<u32> {
        u32::try_from(D).map_err(|_| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension out of range",
            )
        })
    }
}

impl EmbeddingPort for OnnxEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        let test_text = request
            .options
            .test_text
            .unwrap_or_else(|| DEFAULT_TEST_TEXT.to_owned().into_boxed_str());
        let tokenizer = Arc::clone(&self.tokenizer);
        let inputs = self.inputs.clone();
        let output_name = self.output_name.clone();
        let session_pool = Arc::clone(&self.session_pool);
        let dimension_override = self.dimension_override.or(self.model_dimension);

        Box::pin(async move {
            ctx.ensure_not_cancelled("onnx_embedding.detect_dimension")?;
            if let Some(dimension) = dimension_override {
                return Ok(dimension);
            }
            let embeddings =
                Self::run_blocking(&ctx, "onnx_embedding.detect_dimension", move || {
                    let texts = vec![test_text];
                    Self::embed_many_sync(
                        &tokenizer,
                        &inputs,
                        output_name.as_ref(),
                        &session_pool,
                        &texts,
                        dimension_override,
                    )
                })
                .await?;
            let embedding = embeddings.into_iter().next().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "missing ONNX embedding output",
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(embedding.dimension())
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let text = request.text;
        let tokenizer = Arc::clone(&self.tokenizer);
        let inputs = self.inputs.clone();
        let output_name = self.output_name.clone();
        let session_pool = Arc::clone(&self.session_pool);
        let dimension_override = self.dimension_override.or(self.model_dimension);

        Box::pin(async move {
            ctx.ensure_not_cancelled("onnx_embedding.embed")?;
            let embeddings = Self::run_blocking(&ctx, "onnx_embedding.embed", move || {
                let texts = vec![text];
                Self::embed_many_sync(
                    &tokenizer,
                    &inputs,
                    output_name.as_ref(),
                    &session_pool,
                    &texts,
                    dimension_override,
                )
            })
            .await?;
            embeddings.into_iter().next().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "missing ONNX embedding output",
                    ErrorClass::NonRetriable,
                )
            })
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let texts = request.texts;
        let tokenizer = Arc::clone(&self.tokenizer);
        let inputs = self.inputs.clone();
        let output_name = self.output_name.clone();
        let session_pool = Arc::clone(&self.session_pool);
        let dimension_override = self.dimension_override.or(self.model_dimension);

        Box::pin(async move {
            ctx.ensure_not_cancelled("onnx_embedding.embed_batch")?;
            Self::run_blocking(&ctx, "onnx_embedding.embed_batch", move || {
                Self::embed_many_sync(
                    &tokenizer,
                    &inputs,
                    output_name.as_ref(),
                    &session_pool,
                    &texts,
                    dimension_override,
                )
            })
            .await
        })
    }
}

impl<const D: usize> EmbeddingPort for OnnxEmbeddingFixed<D> {
    fn provider(&self) -> &EmbeddingProviderInfo {
        EmbeddingPort::provider(&self.inner)
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("onnx_embedding_fixed.detect_dimension")?;
            Self::expected_dimension()
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let future = self.inner.embed(ctx, request);
        Box::pin(async move {
            let vector = future.await?;
            let fixed = EmbeddingVectorFixed::<D>::try_from(vector)?;
            Ok(EmbeddingVector::from(fixed))
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let future = self.inner.embed_batch(ctx, request);
        Box::pin(async move {
            let vectors = future.await?;
            let fixed = vectors
                .into_iter()
                .map(EmbeddingVectorFixed::<D>::try_from)
                .collect::<Result<Vec<_>>>()?;
            Ok(fixed.into_iter().map(EmbeddingVector::from).collect())
        })
    }
}

#[derive(Debug, Clone)]
struct ModelInputs {
    input_ids: Box<str>,
    attention_mask: Option<Box<str>>,
    token_type_ids: Option<Box<str>>,
}

impl ModelInputs {
    const fn requires_token_type_ids(&self) -> bool {
        self.token_type_ids.is_some()
    }

    const fn requires_attention_mask(&self) -> bool {
        self.attention_mask.is_some()
    }
}

#[derive(Debug, Default)]
struct BatchBuffers {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

impl BatchBuffers {
    fn prepare_inputs<'a>(
        &'a mut self,
        encodings: &[tokenizers::Encoding],
        inputs: &ModelInputs,
    ) -> Result<PreparedInputs<'a>> {
        let batch_size = encodings.len();
        if batch_size == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding input must be non-empty",
            ));
        }

        let sequence_length = encodings
            .first()
            .map_or(0, |encoding| encoding.get_ids().len());
        if sequence_length == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "tokenization produced empty sequence",
            ));
        }

        let total = batch_size.saturating_mul(sequence_length);
        self.input_ids.clear();
        self.attention_mask.clear();
        self.token_type_ids.clear();
        self.input_ids.reserve(total);
        self.attention_mask.reserve(total);
        if inputs.requires_token_type_ids() {
            self.token_type_ids.reserve(total);
        }

        for encoding in encodings {
            let ids_slice = encoding.get_ids();
            let mask_slice = encoding.get_attention_mask();
            if ids_slice.len() != sequence_length || mask_slice.len() != sequence_length {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "tokenizer padding produced inconsistent lengths",
                    ErrorClass::NonRetriable,
                ));
            }
            self.input_ids
                .extend(ids_slice.iter().map(|value| i64::from(*value)));
            self.attention_mask
                .extend(mask_slice.iter().map(|value| i64::from(*value)));

            if inputs.requires_token_type_ids() {
                let type_slice = encoding.get_type_ids();
                if type_slice.is_empty() {
                    self.token_type_ids
                        .extend(std::iter::repeat_n(0, sequence_length));
                } else {
                    if type_slice.len() != sequence_length {
                        return Err(ErrorEnvelope::unexpected(
                            ErrorCode::internal(),
                            "token type ids length mismatch",
                            ErrorClass::NonRetriable,
                        ));
                    }
                    self.token_type_ids
                        .extend(type_slice.iter().map(|value| i64::from(*value)));
                }
            }
        }

        Ok(PreparedInputs {
            batch_size,
            sequence_length,
            input_ids: &self.input_ids,
            attention_mask: &self.attention_mask,
            token_type_ids: if inputs.requires_token_type_ids() {
                Some(&self.token_type_ids)
            } else {
                None
            },
        })
    }
}

#[derive(Debug)]
struct PreparedInputs<'a> {
    batch_size: usize,
    sequence_length: usize,
    input_ids: &'a [i64],
    attention_mask: &'a [i64],
    token_type_ids: Option<&'a [i64]>,
}

fn build_session_inputs<'a>(
    inputs: &ModelInputs,
    batch_size: usize,
    sequence_length: usize,
    input_ids: &'a [i64],
    attention_mask: &'a [i64],
    token_type_ids: Option<&'a [i64]>,
) -> Result<SessionInputs<'a, 'a>> {
    let shape = [batch_size, sequence_length];
    let ids_tensor = TensorRef::from_array_view((shape, input_ids))
        .map_err(map_ort_error("onnx_input_tensor_failed"))?;
    let mut session_inputs: Vec<(Cow<'a, str>, SessionInputValue<'a>)> =
        vec![(Cow::Owned(inputs.input_ids.to_string()), ids_tensor.into())];

    if inputs.requires_attention_mask() {
        let mask_tensor = TensorRef::from_array_view((shape, attention_mask))
            .map_err(map_ort_error("onnx_input_tensor_failed"))?;
        if let Some(name) = inputs.attention_mask.as_ref() {
            session_inputs.push((Cow::Owned(name.to_string()), mask_tensor.into()));
        }
    }

    if inputs.requires_token_type_ids() {
        let type_data = token_type_ids.ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "token type ids missing",
                ErrorClass::NonRetriable,
            )
        })?;
        let types_tensor = TensorRef::from_array_view((shape, type_data))
            .map_err(map_ort_error("onnx_input_tensor_failed"))?;
        if let Some(name) = inputs.token_type_ids.as_ref() {
            session_inputs.push((Cow::Owned(name.to_string()), types_tensor.into()));
        }
    }

    Ok(session_inputs.into())
}

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    model_max_length: Option<usize>,
    pad_token: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ModelConfig {
    hidden_size: Option<u64>,
    max_position_embeddings: Option<usize>,
    pad_token_id: Option<u32>,
}

fn normalize_model_dir(model_dir: &Path) -> Result<PathBuf> {
    if !model_dir.exists() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "ONNX model directory does not exist",
        )
        .with_metadata("path", model_dir.to_string_lossy().to_string()));
    }
    Ok(model_dir.to_path_buf())
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

fn resolve_model_path(model_dir: &Path, override_name: Option<&str>) -> Result<PathBuf> {
    if let Some(name) = override_name {
        let candidate = model_dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "ONNX model file not found",
        )
        .with_metadata("path", candidate.to_string_lossy().to_string()));
    }

    let nested = model_dir.join("onnx").join(DEFAULT_MODEL_FILE);
    if nested.exists() {
        return Ok(nested);
    }
    let root = model_dir.join(DEFAULT_MODEL_FILE);
    if root.exists() {
        return Ok(root);
    }

    Err(
        ErrorEnvelope::expected(ErrorCode::invalid_input(), "ONNX model file not found")
            .with_metadata("path", model_dir.to_string_lossy().to_string()),
    )
}

fn resolve_tokenizer_path(model_dir: &Path, override_name: Option<&str>) -> Result<PathBuf> {
    let file_name = override_name.unwrap_or(DEFAULT_TOKENIZER_FILE);
    let path = model_dir.join(file_name);
    if !path.exists() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "tokenizer file not found",
        )
        .with_metadata("path", path.to_string_lossy().to_string()));
    }
    Ok(path)
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("failed to load tokenizer: {error}"),
        )
        .with_metadata("path", path.to_string_lossy().to_string())
    })
}

fn read_tokenizer_config(model_dir: &Path) -> Result<TokenizerConfig> {
    let path = model_dir.join("tokenizer_config.json");
    if !path.exists() {
        return Ok(TokenizerConfig::default());
    }
    let content = std::fs::read_to_string(&path).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::io(),
            format!("failed to read tokenizer config: {error}"),
        )
        .with_metadata("path", path.to_string_lossy().to_string())
    })?;
    serde_json::from_str(&content).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("invalid tokenizer config json: {error}"),
        )
        .with_metadata("path", path.to_string_lossy().to_string())
    })
}

fn read_model_config(model_dir: &Path) -> Result<ModelConfig> {
    let path = model_dir.join("config.json");
    if !path.exists() {
        return Ok(ModelConfig::default());
    }
    let content = std::fs::read_to_string(&path).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::io(),
            format!("failed to read model config: {error}"),
        )
        .with_metadata("path", path.to_string_lossy().to_string())
    })?;
    serde_json::from_str(&content).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("invalid model config json: {error}"),
        )
        .with_metadata("path", path.to_string_lossy().to_string())
    })
}

fn resolve_max_length(
    tokenizer_config: &TokenizerConfig,
    model_config: &ModelConfig,
) -> Option<usize> {
    match (
        tokenizer_config.model_max_length,
        model_config.max_position_embeddings,
    ) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn configure_tokenizer(
    mut tokenizer: Tokenizer,
    tokenizer_config: &TokenizerConfig,
    model_config: &ModelConfig,
    max_length: Option<usize>,
) -> Result<Tokenizer> {
    if tokenizer.get_padding().is_none() {
        let pad_token = tokenizer_config
            .pad_token
            .clone()
            .unwrap_or_else(|| "[PAD]".to_string());
        let pad_id = model_config
            .pad_token_id
            .or_else(|| tokenizer.token_to_id(pad_token.as_str()))
            .unwrap_or(0);
        let padding = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        };
        tokenizer.with_padding(Some(padding));
    }

    if tokenizer.get_truncation().is_none()
        && let Some(max_length) = max_length
    {
        let truncation = TruncationParams {
            max_length,
            strategy: TruncationStrategy::OnlyFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        };
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    format!("invalid tokenizer truncation params: {error}"),
                )
            })?;
    }

    Ok(tokenizer)
}

fn resolve_inputs(inputs: &[ort::value::Outlet]) -> Result<ModelInputs> {
    let mut input_ids = None;
    let mut attention_mask = None;
    let mut token_type_ids = None;

    for outlet in inputs {
        let name = outlet.name();
        let name_lower = name.to_ascii_lowercase();
        if name_lower.contains(DEFAULT_INPUT_IDS) {
            input_ids = Some(name.to_owned().into_boxed_str());
        } else if name_lower.contains(DEFAULT_ATTENTION_MASK) {
            attention_mask = Some(name.to_owned().into_boxed_str());
        } else if name_lower.contains("token_type") {
            token_type_ids = Some(name.to_owned().into_boxed_str());
        }
    }

    let input_ids = input_ids.ok_or_else(|| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "ONNX model missing input_ids input",
        )
    })?;

    Ok(ModelInputs {
        input_ids,
        attention_mask,
        token_type_ids,
    })
}

fn map_pooled_embeddings(
    shape: &ort::tensor::Shape,
    data: &[f32],
    batch_size: usize,
    expected_dimension: Option<u32>,
) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 2 {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "expected 2D pooled output",
            ErrorClass::NonRetriable,
        ));
    }
    let hidden_dim = shape.get(1).copied().ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "pooled output missing dimension",
            ErrorClass::NonRetriable,
        )
    })?;
    let hidden = if hidden_dim > 0 {
        usize::try_from(hidden_dim).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding dimension overflow",
                ErrorClass::NonRetriable,
            )
        })?
    } else {
        data.len().checked_div(batch_size).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "invalid pooled output shape",
                ErrorClass::NonRetriable,
            )
        })?
    };

    validate_dimension(hidden, expected_dimension)?;
    if data.len() != batch_size.saturating_mul(hidden) {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "pooled output shape mismatch",
            ErrorClass::NonRetriable,
        ));
    }

    Ok(data.chunks_exact(hidden).map(<[f32]>::to_vec).collect())
}

fn map_sequence_embeddings(
    shape: &ort::tensor::Shape,
    data: &[f32],
    batch_size: usize,
    sequence_length: usize,
    attention_mask: &[i64],
    expected_dimension: Option<u32>,
) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 3 {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "expected 3D sequence output",
            ErrorClass::NonRetriable,
        ));
    }
    let hidden_dim = shape.get(2).copied().ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "sequence output missing dimension",
            ErrorClass::NonRetriable,
        )
    })?;
    let hidden = if hidden_dim > 0 {
        usize::try_from(hidden_dim).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding dimension overflow",
                ErrorClass::NonRetriable,
            )
        })?
    } else {
        let denom = batch_size.saturating_mul(sequence_length);
        data.len().checked_div(denom).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "invalid sequence output shape",
                ErrorClass::NonRetriable,
            )
        })?
    };

    validate_dimension(hidden, expected_dimension)?;
    let expected = batch_size
        .saturating_mul(sequence_length)
        .saturating_mul(hidden);
    if data.len() != expected {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "sequence output shape mismatch",
            ErrorClass::NonRetriable,
        ));
    }

    let mut embeddings = Vec::with_capacity(batch_size);
    for batch in 0..batch_size {
        let mut vector = vec![0.0f32; hidden];
        let start = batch.saturating_mul(sequence_length);
        let end = batch.saturating_add(1).saturating_mul(sequence_length);
        let mask_slice = attention_mask.get(start..end).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "attention mask shape mismatch",
                ErrorClass::NonRetriable,
            )
        })?;
        let mut denom = 0.0f32;
        for (token_idx, &mask) in mask_slice.iter().enumerate() {
            if mask == 0 {
                continue;
            }
            denom += 1.0;
            let token_start = (batch * sequence_length + token_idx) * hidden;
            let token_end = token_start.saturating_add(hidden);
            let token_slice = data.get(token_start..token_end).ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "sequence output shape mismatch",
                    ErrorClass::NonRetriable,
                )
            })?;
            for (value, token_value) in vector.iter_mut().zip(token_slice.iter()) {
                *value += token_value;
            }
        }
        if denom > 0.0 {
            for value in &mut vector {
                *value /= denom;
            }
        }
        embeddings.push(vector);
    }

    Ok(embeddings)
}

fn normalize_embedding(vector: Vec<f32>) -> EmbeddingVector {
    let mut vector = vector;
    let mut norm = 0.0f32;
    for value in &vector {
        norm += value * value;
    }
    if norm > 0.0 {
        let denom = norm.sqrt();
        for value in &mut vector {
            *value /= denom;
        }
    }
    let vector = Arc::<[f32]>::from(vector);
    EmbeddingVector::new(vector)
}

fn validate_dimension(hidden: usize, expected_dimension: Option<u32>) -> Result<()> {
    if let Some(expected) = expected_dimension {
        let expected_usize = usize::try_from(expected).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding dimension overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        if hidden != expected_usize {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension mismatch",
            )
            .with_metadata("expected", expected.to_string())
            .with_metadata("actual", hidden.to_string()));
        }
    }
    Ok(())
}

fn map_ort_error(code: &'static str) -> impl FnOnce(ort::Error) -> ErrorEnvelope {
    move |error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("embedding", code),
            format!("ONNX runtime error: {error}"),
            ErrorClass::NonRetriable,
        )
    }
}
