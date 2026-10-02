// Copyright (c) 2026 OpenAgenet contributors
//
// Initial author: JINLIANG XU
// Email: jlxufly@gmail.com

use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{Datelike, Utc};
use futures::TryStreamExt;
use jieba_rs::Jieba;
use oan_core::{
    CryptoSuite, DidDocument, ImplementationLink, OanMetadata, ProtocolBinding,
    ResourceDescription, ResourceType, ServiceEndpoint, SubjectType, VerificationMethod,
};
use oan_credentials::OanIdentity;
use oan_crypto::{
    did_document_signature_input, generate_ed25519_keypair, hash_json_with_suite, public_key_jwk,
    public_key_multibase, sha256_hex, sign_bytes_multibase, verify_did_document_proof,
    verify_did_document_proof_standard_blocking, SigningKey, VerifyingKey,
};
use oan_package::{
    hash_resource_metadata_with_suite, ResourceMetadata, ResourcePackage, ResourcePackageClaims,
    RootProof,
};
use oan_protocol::{
    HealthResponse, ResourceDiscoveryCandidate, ResourceDiscoveryQuery, ResourceDiscoveryResponse,
};
use oan_semantic_recommender::{
    DiscoverySuggestionContext, DiscoverySuggestionInput, SemanticRecommender,
};
use oan_storage::{DatabaseBackend, DatabaseConfig, JsonStore, PostgresJsonStore, SqliteJsonStore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{Postgres, QueryBuilder, Row, Sqlite};
#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env, fs,
    io::{BufWriter, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::{Duration as StdDuration, Instant},
};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration as TokioDuration};
use tower_http::cors::{AllowHeaders, AllowOrigin, CorsLayer};

const DISCOVERY_SYNC_STATE_TABLE: &str = "discovery_sync_state";
const DISCOVERY_PACKAGE_TABLE: &str = "discovery_packages";
const DISCOVERY_REJECTED_TABLE: &str = "discovery_rejected_packages";
const DISCOVERY_SEMANTIC_INDEX_TABLE: &str = "discovery_semantic_index";
const DISCOVERY_INTENT_INDEX_TABLE: &str = "discovery_intent_index";
const DISCOVERY_INDEX_STATS_CACHE_TTL_MS: u64 = 500;
const DISCOVERY_CDN_CURSOR_KEY: &str = "cdn_publication_cursor";
const DISCOVERY_QUERY_STATS_TABLE: &str = "discovery_query_stats";
const DEFAULT_SEMANTIC_DIMENSION: usize = 64;
const SEMANTIC_ALIASES_JSON: &str = include_str!("semantic_aliases.v1.json");
const REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE: &str = "resource_package_unavailable";
const REJECT_STATUS_LATE: &str = "late";
const REJECT_STATUS_DEAD: &str = "dead";
const LATE_PACKAGE_RETRY_INTERVAL_SECONDS: u64 = 120;
const LATE_PACKAGE_MAX_RETRY_COUNT: u64 = 3;
const LATE_PACKAGE_MAX_ITEMS: usize = 100;
const DISCOVERY_DEFAULT_PAGE_SIZE: u32 = 100;
const DISCOVERY_MAX_PAGE_SIZE: u32 = 500;
const DISCOVERY_SEMANTIC_REBUILD_PAGE_SIZE: u32 = 100;
const DISCOVERY_INDEX_PACKAGE_BYTES: usize = 8 * 1024 * 1024;
const DISCOVERY_INDEX_PAGE_BYTES: usize = 32 * 1024 * 1024;
#[cfg(not(test))]
const LATE_PACKAGE_IDLE_STOP_CYCLES: u64 = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LatePackageRetryExit {
    Idle,
    Suspended,
}

fn should_rewake_late_package_retry(exit: LatePackageRetryExit, has_late: bool) -> bool {
    has_late && exit != LatePackageRetryExit::Suspended
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DiscoverySyncRequest {
    #[serde(rename = "maxPublications", default)]
    max_publications: Option<usize>,
    #[serde(rename = "cursorHint", default)]
    cursor_hint: Option<i64>,
    #[serde(default)]
    items: Vec<DiscoveryNotificationItem>,
}

#[derive(Clone, Debug, Deserialize)]
struct DiscoveryNotificationItem {
    #[serde(rename = "resourceDid")]
    resource_did: String,
    // Kept in the notification contract for audit/compatibility. The discovery
    // node indexes the current CDN package and validates the package itself.
    #[allow(dead_code)]
    #[serde(rename = "packageVersion")]
    package_version: String,
    #[serde(rename = "publicationCursor")]
    publication_cursor: i64,
    #[allow(dead_code)]
    #[serde(rename = "packageHash")]
    package_hash: String,
    #[allow(dead_code)]
    #[serde(rename = "metadataHash")]
    metadata_hash: String,
    #[allow(dead_code)]
    #[serde(rename = "didDocumentHash")]
    did_document_hash: String,
    #[serde(rename = "resourceType", default)]
    resource_type: Option<String>,
    #[serde(rename = "capabilityTags", default)]
    capability_tags: Vec<String>,
    #[serde(rename = "authorizedDomains", default)]
    authorized_domains: Vec<String>,
}

#[derive(Clone, Debug)]
enum PackageFetchOutcome {
    Found(Box<ResourcePackage>),
    Unavailable(String),
    PermanentFailure(String),
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RejectedPackageBackfillRequest {
    #[serde(rename = "maxItems", default)]
    max_items: Option<usize>,
    #[serde(rename = "resourceDids", default)]
    resource_dids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ResolvedRejectedPackageCleanupRequest {
    #[serde(rename = "dryRun", default = "default_true")]
    dry_run: bool,
    #[serde(rename = "maxItems", default)]
    max_items: Option<usize>,
    #[serde(rename = "resourceDids", default)]
    resource_dids: Vec<String>,
}

impl Default for ResolvedRejectedPackageCleanupRequest {
    fn default() -> Self {
        Self {
            dry_run: true,
            max_items: None,
            resource_dids: vec![],
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct IndexedResourceVisibilityRequest {
    #[serde(rename = "resourceDids")]
    resource_dids: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DiscoveryPageQuery {
    #[serde(rename = "afterCursor", default)]
    after_cursor: Option<i64>,
    #[serde(rename = "afterResourceDid", default)]
    after_resource_did: Option<String>,
    #[serde(rename = "afterKey", default)]
    after_key: Option<String>,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct Config {
    server: ServerConfig,
    #[serde(default)]
    cors: CorsConfig,
    #[serde(default)]
    debug: DebugConfig,
    #[serde(default)]
    stats_report: StatsReportConfig,
    #[serde(default, rename = "semanticSearch")]
    semantic_search: SemanticSearchConfig,
    upstream: UpstreamConfig,
    paths: PathConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct CorsConfig {
    #[serde(default)]
    allowed_origins: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct DebugConfig {
    #[serde(default)]
    export_snapshots: bool,
    #[serde(default = "default_debug_export_interval_ms")]
    export_interval_ms: u64,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            export_snapshots: false,
            export_interval_ms: default_debug_export_interval_ms(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct UpstreamConfig {
    root_endpoint: String,
    #[serde(default)]
    cdn_endpoint: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct PathConfig {
    data_dir: PathBuf,
    index_dir: PathBuf,
    #[serde(default = "default_identity_file")]
    identity_file: PathBuf,
    #[serde(default)]
    database_url: Option<String>,
}

fn default_identity_file() -> PathBuf {
    PathBuf::from("../../data/discovery/identity.json")
}

#[derive(Clone, Debug, Deserialize)]
struct StatsReportConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default = "default_stats_report_interval_seconds")]
    interval_seconds: u64,
}

impl Default for StatsReportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            node_id: None,
            endpoint: None,
            token: None,
            interval_seconds: default_stats_report_interval_seconds(),
        }
    }
}

fn default_debug_export_interval_ms() -> u64 {
    2_000
}

fn default_stats_report_interval_seconds() -> u64 {
    60
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticSearchConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_semantic_engine")]
    engine: String,
    #[serde(default = "default_semantic_fallback")]
    fallback: String,
    #[serde(default = "default_semantic_mode", rename = "defaultMode")]
    default_mode: String,
    #[serde(
        default = "default_semantic_candidate_multiplier",
        rename = "candidateMultiplier"
    )]
    candidate_multiplier: u32,
    #[serde(default = "default_semantic_max_candidates", rename = "maxCandidates")]
    max_candidates: u32,
    #[serde(
        default = "default_semantic_explain_enabled",
        rename = "explainEnabled"
    )]
    explain_enabled: bool,
    #[serde(default)]
    embedding: SemanticEmbeddingConfig,
    #[serde(default)]
    filters: SemanticFilterConfig,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            engine: default_semantic_engine(),
            fallback: default_semantic_fallback(),
            default_mode: default_semantic_mode(),
            candidate_multiplier: default_semantic_candidate_multiplier(),
            max_candidates: default_semantic_max_candidates(),
            explain_enabled: default_semantic_explain_enabled(),
            embedding: SemanticEmbeddingConfig::default(),
            filters: SemanticFilterConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticEmbeddingConfig {
    #[serde(default = "default_semantic_embedding_provider")]
    provider: String,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default, rename = "healthEndpoint")]
    health_endpoint: Option<String>,
    #[serde(default, rename = "modelPath")]
    model_path: Option<String>,
    #[serde(default = "default_semantic_embedding_model", rename = "modelName")]
    model_name: String,
    #[serde(default, rename = "modelVersion")]
    model_version: Option<String>,
    #[serde(default, rename = "strictModelVersion")]
    strict_model_version: bool,
    #[serde(default, rename = "dtype")]
    dtype: Option<String>,
    #[serde(default = "default_semantic_dimension")]
    dimension: usize,
    #[serde(
        default = "default_semantic_embedding_timeout_ms",
        rename = "timeoutMs"
    )]
    timeout_ms: u64,
    #[serde(
        default = "default_semantic_embedding_batch_size",
        rename = "batchSize"
    )]
    batch_size: usize,
    #[serde(
        default = "default_semantic_embedding_max_input_chars",
        rename = "maxInputChars"
    )]
    max_input_chars: usize,
}

impl Default for SemanticEmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_semantic_embedding_provider(),
            endpoint: None,
            health_endpoint: None,
            model_path: None,
            model_name: default_semantic_embedding_model(),
            model_version: None,
            strict_model_version: false,
            dtype: None,
            dimension: default_semantic_dimension(),
            timeout_ms: default_semantic_embedding_timeout_ms(),
            batch_size: default_semantic_embedding_batch_size(),
            max_input_chars: default_semantic_embedding_max_input_chars(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticFilterConfig {
    #[serde(
        default = "default_semantic_resource_type_filter",
        rename = "resourceType"
    )]
    resource_type: String,
    #[serde(
        default = "default_semantic_capability_tags_filter",
        rename = "capabilityTags"
    )]
    capability_tags: String,
    #[serde(default = "default_semantic_protocol_filter")]
    protocol: String,
}

impl Default for SemanticFilterConfig {
    fn default() -> Self {
        Self {
            resource_type: default_semantic_resource_type_filter(),
            capability_tags: default_semantic_capability_tags_filter(),
            protocol: default_semantic_protocol_filter(),
        }
    }
}

fn default_semantic_engine() -> String {
    "pgvector".to_owned()
}

fn default_semantic_fallback() -> String {
    "keyword".to_owned()
}

fn default_semantic_mode() -> String {
    "hybrid".to_owned()
}

fn default_semantic_candidate_multiplier() -> u32 {
    8
}

fn default_semantic_max_candidates() -> u32 {
    500
}

fn default_semantic_explain_enabled() -> bool {
    true
}

fn default_semantic_embedding_provider() -> String {
    "deterministic-local".to_owned()
}

fn default_semantic_embedding_model() -> String {
    "oan-deterministic-local-v1".to_owned()
}

fn default_semantic_dimension() -> usize {
    DEFAULT_SEMANTIC_DIMENSION
}

fn default_semantic_embedding_timeout_ms() -> u64 {
    2_000
}

fn default_semantic_embedding_batch_size() -> usize {
    32
}

fn default_semantic_embedding_max_input_chars() -> usize {
    6_000
}

fn default_semantic_evaluation_package_version() -> String {
    "1".to_owned()
}

fn default_semantic_resource_type_filter() -> String {
    "hard".to_owned()
}

fn default_semantic_capability_tags_filter() -> String {
    "hard".to_owned()
}

fn default_semantic_protocol_filter() -> String {
    "hard".to_owned()
}

#[derive(Clone)]
struct AppState {
    data: JsonStore,
    index: JsonStore,
    config: Config,
    did: String,
    sqlite: Option<SqliteJsonStore>,
    postgres: Option<PostgresJsonStore>,
    semantic: SemanticRuntimeState,
    recommender: Arc<SemanticRecommender>,
    client: reqwest::Client,
    resource_sync_lock: Arc<Mutex<()>>,
    #[cfg(not(test))]
    late_package_retry_running: Arc<AtomicBool>,
    index_stats_cache: Arc<Mutex<Option<CachedIndexStats>>>,
}

#[derive(Clone, Debug)]
struct SemanticRuntimeState {
    enabled: bool,
    healthy: bool,
    backend: String,
    fallback_reason: Option<String>,
    embedding_model: String,
    embedding_version: String,
    embedding_dtype: Option<String>,
    dimension: usize,
}

impl SemanticRuntimeState {
    fn disabled(reason: impl Into<String>, config: &SemanticSearchConfig) -> Self {
        Self {
            enabled: false,
            healthy: false,
            backend: config.engine.clone(),
            fallback_reason: Some(reason.into()),
            embedding_model: config.embedding.model_name.clone(),
            embedding_version: semantic_embedding_version(config),
            embedding_dtype: config.embedding.dtype.clone(),
            dimension: config.embedding.dimension,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiscoverySearchEngine {
    PgVector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticRebuildStage {
    Both,
    Context,
    Intent,
}

impl SemanticRebuildStage {
    fn from_arg(value: &str) -> Result<Self> {
        match value {
            "both" => Ok(Self::Both),
            "context" => Ok(Self::Context),
            "intent" => Ok(Self::Intent),
            value => Err(anyhow!("unsupported_semantic_rebuild_stage:{value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::Context => "context",
            Self::Intent => "intent",
        }
    }

    fn includes_context(self) -> bool {
        matches!(self, Self::Both | Self::Context)
    }

    fn includes_intent(self) -> bool {
        matches!(self, Self::Both | Self::Intent)
    }
}

impl DiscoverySearchEngine {
    fn from_config(config: &SemanticSearchConfig) -> Result<Self> {
        match config.engine.as_str() {
            "pgvector" => Ok(Self::PgVector),
            "grail" => Err(anyhow!("grail_engine_not_implemented")),
            value => Err(anyhow!("unsupported_semantic_engine:{value}")),
        }
    }

    async fn initialize(
        &self,
        postgres: &PostgresJsonStore,
        config: &SemanticSearchConfig,
    ) -> Result<()> {
        match self {
            Self::PgVector => initialize_semantic_postgres(postgres, config).await,
        }
    }

    async fn upsert_batch(
        &self,
        state: &AppState,
        projected: &[DiscoveryProjectedPackage<'_>],
        stage: SemanticRebuildStage,
    ) -> Result<()> {
        match self {
            Self::PgVector => upsert_pgvector_semantic_index_batch(state, projected, stage).await,
        }
    }

    async fn query(
        &self,
        state: &AppState,
        query: &ResourceDiscoveryQuery,
    ) -> Result<Option<Vec<SemanticQueryHit>>> {
        match self {
            Self::PgVector => query_pgvector_semantic_index(state, query).await,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingProviderKind {
    DeterministicLocal,
    HttpEmbedding,
}

impl EmbeddingProviderKind {
    fn from_config(config: &SemanticEmbeddingConfig) -> Result<Self> {
        match config.provider.as_str() {
            "deterministic-local" => Ok(Self::DeterministicLocal),
            "http-embedding" => Ok(Self::HttpEmbedding),
            "local" | "local-model" => Err(anyhow!("local_embedding_provider_not_wired")),
            value => Err(anyhow!("unsupported_embedding_provider:{value}")),
        }
    }
}

#[derive(Debug, Serialize)]
struct HttpEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HttpEmbeddingResponse {
    model: String,
    #[serde(rename = "embeddingVersion", alias = "embedding_version")]
    embedding_version: String,
    #[serde(default)]
    dtype: Option<String>,
    dimension: usize,
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct HttpEmbeddingHealthResponse {
    model: String,
    #[serde(rename = "embeddingVersion", alias = "embedding_version")]
    embedding_version: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
    dimension: usize,
    #[serde(default)]
    ready: bool,
}

#[derive(Clone, Debug)]
struct CachedIndexStats {
    captured_at: Instant,
    body: Value,
}

#[derive(Clone, Debug)]
struct DiscoveryPackageProjection {
    resource_type: String,
    lifecycle_state: String,
    capability_tags: Vec<String>,
    protocols: Vec<String>,
    service_endpoints: Vec<String>,
    tag_text: String,
    search_text: String,
}

struct DiscoveryProjectedPackage<'a> {
    cursor: i64,
    package: &'a ResourcePackage,
    package_json: Value,
    projection: DiscoveryPackageProjection,
}

#[derive(Clone, Debug)]
struct CursorPage<T> {
    items: Vec<T>,
    next_cursor: i64,
    next_resource_did: Option<String>,
    has_more: bool,
}

#[derive(Clone, Debug)]
struct KeyPage<T> {
    items: Vec<T>,
    next_key: Option<String>,
    has_more: bool,
}

#[derive(Clone, Debug)]
struct SearchDocument {
    resource_did: String,
    cursor: i64,
    package_version: String,
    did_document_hash: String,
    metadata_hash: String,
    package_hash: String,
    resource_type: String,
    lifecycle_state: String,
    capability_tags: Vec<String>,
    protocols: Vec<String>,
    service_endpoints: Vec<String>,
    name: String,
    description: String,
    examples: Vec<String>,
    use_cases: Vec<String>,
    tag_text: String,
    search_text: String,
    semantic_source_hash: String,
}

#[derive(Clone, Debug)]
struct IntentDocument {
    resource_did: String,
    intent_id: String,
    intent_kind: String,
    intent_text: String,
    resource_type: String,
    lifecycle_state: String,
    semantic_source_hash: String,
}

#[derive(Clone, Debug)]
struct SemanticQueryHit {
    package: ResourcePackage,
    semantic_score: f32,
    bm25_score: f32,
    tag_score: f32,
    context_score: f32,
    intent_score: f32,
    structured_score: f32,
    final_score: f32,
    matched_intent: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SemanticAliasFile {
    groups: Vec<SemanticAliasGroup>,
}

#[derive(Debug, Deserialize)]
struct SemanticAliasGroup {
    canonical: String,
    terms: Vec<String>,
}

#[derive(Clone, Debug)]
struct SemanticAliasCatalog {
    phrase_to_canonical: Vec<(String, String)>,
    canonical_to_terms: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug)]
struct SemanticQueryProjection {
    expanded_tokens: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct SemanticSearchDiagnostics {
    enabled: bool,
    healthy: bool,
    backend: String,
    embedding_model: String,
    embedding_version: String,
    embedding_dtype: Option<String>,
    dimension: usize,
    fallback_used: bool,
    fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DiscoveryQueryMetrics {
    backend: String,
    candidate_count: usize,
    returned_count: usize,
    elapsed_ms: u128,
    used_indexed_prefilter: bool,
}

#[derive(Debug, Serialize)]
struct SemanticRebuildReport {
    stage: String,
    phase: String,
    backend: String,
    semantic_enabled: bool,
    package_count: usize,
    context_indexed_count: usize,
    intent_indexed_count: usize,
    skipped_packages: usize,
    recovered_packages: usize,
    retries: usize,
    last_cursor: Option<i64>,
    safe_to_delete_old_indexes: bool,
    skipped_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct SemanticRebuildProgressRecord {
    #[serde(default)]
    stage: String,
    #[serde(default)]
    phase: String,
    #[serde(default)]
    last_cursor: Option<i64>,
    #[serde(rename = "lastResourceDid", default)]
    last_resource_did: Option<String>,
    #[serde(default)]
    processed: usize,
    #[serde(default)]
    total: usize,
    #[serde(default)]
    skipped_packages: usize,
    #[serde(default)]
    retries: usize,
    #[serde(default)]
    batch_size: usize,
    #[serde(default)]
    updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct SemanticRebuildSkipRecord {
    #[serde(default)]
    stage: String,
    #[serde(rename = "resourceDid", default)]
    resource_did: String,
    #[serde(default)]
    cursor: i64,
    #[serde(default)]
    reason: String,
    #[serde(rename = "updatedAt", default)]
    updated_at: String,
}

#[derive(Debug, Deserialize)]
struct SemanticEvaluationSuite {
    suite: String,
    #[serde(default, rename = "seedPackages")]
    seed_packages: Vec<ResourcePackage>,
    #[serde(default, rename = "seedResources")]
    seed_resources: Vec<SemanticEvaluationSeedResource>,
    cases: Vec<SemanticEvaluationCase>,
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticEvaluationSeedResource {
    #[serde(rename = "resourceDid")]
    resource_did: String,
    #[serde(rename = "resourceType")]
    resource_type: ResourceType,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "capabilityDescription", default)]
    capability_description: String,
    #[serde(rename = "capabilityTags", default)]
    capability_tags: Vec<String>,
    #[serde(rename = "authorizedDomains", default)]
    authorized_domains: Vec<String>,
    #[serde(default)]
    protocols: Vec<String>,
    #[serde(rename = "serviceEndpoint", default)]
    service_endpoint: Option<String>,
    #[serde(rename = "useCaseExamples", default)]
    use_case_examples: Vec<String>,
    #[serde(default)]
    examples: Vec<Value>,
    #[serde(
        default = "default_semantic_evaluation_package_version",
        rename = "packageVersion"
    )]
    package_version: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticEvaluationCase {
    id: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    description: Option<String>,
    query: ResourceDiscoveryQuery,
    #[serde(rename = "expectedDids")]
    expected_dids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SemanticEvaluationReport {
    suite: String,
    case_count: usize,
    seed_package_count: usize,
    recall_at_5: f32,
    recall_at_10: f32,
    mrr_at_10: f32,
    cases: Vec<SemanticEvaluationCaseReport>,
}

#[derive(Debug, Serialize)]
struct SemanticEvaluationCaseReport {
    id: String,
    language: Option<String>,
    description: Option<String>,
    passed: bool,
    reciprocal_rank: f32,
    returned_dids: Vec<String>,
    expected_dids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn internal(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

fn parse_semantic_cli_args(args: &[String]) -> Result<(String, SemanticRebuildStage, bool)> {
    let mut config_path = "services/discovery-node/config.example.toml".to_owned();
    let mut stage = SemanticRebuildStage::Both;
    let mut reset = false;
    let mut index = 1usize;
    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "--reset" => {
                reset = true;
                index += 1;
            }
            "--stage" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| anyhow!("missing_semantic_rebuild_stage"))?;
                stage = SemanticRebuildStage::from_arg(value)?;
                index += 2;
            }
            value if value.starts_with("--stage=") => {
                let value = value.split_once('=').map(|(_, rhs)| rhs).unwrap_or("");
                stage = SemanticRebuildStage::from_arg(value)?;
                index += 1;
            }
            value if value.starts_with("--") => {
                index += 1;
            }
            value => {
                config_path = value.to_owned();
                index += 1;
            }
        }
    }
    Ok((config_path, stage, reset))
}

fn semantic_rebuild_progress_key(stage: SemanticRebuildStage) -> String {
    format!("semantic_rebuild_progress:{}", stage.as_str())
}

fn semantic_rebuild_skip_prefix(stage: Option<SemanticRebuildStage>) -> String {
    match stage {
        Some(stage) => format!("semantic_rebuild_skip:{}:", stage.as_str()),
        None => "semantic_rebuild_skip:".to_owned(),
    }
}

fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()?
        .block_on(async { tokio::spawn(async_main()).await? })
}

async fn async_main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args
        .first()
        .is_some_and(|value| value == "semantic-rebuild")
    {
        let (config_path, stage, reset) = parse_semantic_cli_args(&args)?;
        let state = build_app_state(load_config(config_path)?).await?;
        let report = rebuild_semantic_indexes(&state, reset, stage).await?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if args
        .first()
        .is_some_and(|value| value == "semantic-backfill-skipped")
    {
        let (config_path, stage, _) = parse_semantic_cli_args(&args)?;
        let state = build_app_state(load_config(config_path)?).await?;
        let report = backfill_skipped_semantic_indexes(&state, stage).await?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if args
        .first()
        .is_some_and(|value| value == "semantic-evaluate")
    {
        let config_path = args
            .get(1)
            .cloned()
            .unwrap_or_else(|| "services/discovery-node/config.example.toml".to_owned());
        let suite_path = args.get(2).cloned().unwrap_or_else(|| {
            "services/discovery-node/evaluation/real-resource-smoke.v1.json".to_owned()
        });
        let state = build_app_state(load_config(config_path)?).await?;
        let report = evaluate_semantic_discovery(&state, &PathBuf::from(suite_path)).await?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if args
        .first()
        .is_some_and(|value| value == "semantic-rebuild-evaluate")
    {
        let config_path = args
            .get(1)
            .cloned()
            .unwrap_or_else(|| "services/discovery-node/config.example.toml".to_owned());
        let suite_path = args.get(2).cloned().unwrap_or_else(|| {
            "services/discovery-node/evaluation/real-resource-smoke.v1.json".to_owned()
        });
        let state = build_app_state(load_config(config_path)?).await?;
        let suite_path = PathBuf::from(suite_path);
        let seed_count = seed_semantic_evaluation_packages(&state, &suite_path).await?;
        let rebuild = rebuild_semantic_indexes(&state, true, SemanticRebuildStage::Both).await?;
        let evaluation = evaluate_semantic_discovery(&state, &suite_path).await?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "seedPackageCount": seed_count,
                "rebuild": rebuild,
                "evaluation": evaluation
            }))?
        );
        return Ok(());
    }
    let config_path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "services/discovery-node/config.example.toml".to_owned());
    let config = load_config(config_path)?;
    validate_stats_report_config(&config.stats_report)?;
    let state = build_app_state(config.clone()).await?;

    let app = Router::new()
        .route("/health", get(health))
        .route("/discovery/did", get(discovery_did_document))
        .route("/routes/{did}", get(route_lookup))
        .route("/discovery/status", get(api_status))
        .route("/discovery/root-authorization", get(api_root_authorization))
        .route("/discovery/authorized-domains", get(api_authorized_domains))
        .route(
            "/discovery/resources/sync-authorized",
            post(sync_resources_from_authorized_summary),
        )
        .route("/discovery/sync/history", get(api_sync_history))
        .route("/discovery/index/stats", get(api_index_stats))
        .route("/discovery/query/stats", get(api_query_stats))
        .route("/discovery/index/resources", get(api_index_resources))
        .route(
            "/discovery/index/resources/visibility",
            post(api_index_resource_visibility),
        )
        .route(
            "/discovery/index/resources/{did}",
            get(api_index_resource_detail),
        )
        .route("/discovery/resources/query", post(resource_query))
        .route("/discovery/query/suggestions", post(api_query_suggestions))
        .route("/discovery/query/explain", post(api_query_explain))
        .route("/discovery/rejected-packages", get(api_rejected_packages))
        .route(
            "/discovery/rejected-packages/backfill",
            post(api_backfill_rejected_packages),
        )
        .route(
            "/discovery/rejected-packages/resolved/audit",
            post(api_audit_resolved_rejected_packages),
        )
        .route("/discovery/capability-tree", get(api_capability_tree))
        .layer(build_cors_layer(&config.cors)?)
        .with_state(state.clone());

    if (state.sqlite.is_some() || state.postgres.is_some()) && state.config.debug.export_snapshots {
        let debug_state = state.clone();
        tokio::spawn(async move {
            discovery_debug_export_loop(debug_state).await;
        });
    }
    if state.postgres.is_some() && semantic_search_available(&state) {
        let semantic_state = state.clone();
        tokio::spawn(async move {
            semantic_index_compensation_loop(semantic_state).await;
        });
    }
    if state.config.stats_report.enabled {
        let report_state = state.clone();
        tokio::spawn(async move {
            discovery_stats_report_loop(report_state).await;
        });
    }
    #[cfg(not(test))]
    wake_late_package_retry_task(state.clone());

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    println!("discovery-node listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn build_app_state(config: Config) -> Result<AppState> {
    let identity: OanIdentity = JsonStore::new(".").read(&config.paths.identity_file)?;
    identity
        .validate_data_integrity()
        .await
        .map_err(|err| anyhow!("invalid OAN Identity: {err}"))?;
    let did_doc: DidDocument = identity.did_document.clone();
    #[cfg(not(test))]
    did_doc
        .validate_infrastructure_profile(ResourceType::DiscoveryNode)
        .map_err(|error| anyhow!("invalid discovery DID document profile: {error}"))?;
    // Public projection used by existing discovery paths; identity.json remains authoritative.
    JsonStore::new(&config.paths.data_dir).write("did-document.json", &did_doc)?;
    let mut semantic = SemanticRuntimeState::disabled("semantic_disabled", &config.semantic_search);
    let (sqlite, postgres) = match config.paths.database_url.as_deref() {
        Some(url) if !url.is_empty() => {
            let database = DatabaseConfig::parse(url)?;
            match database.backend() {
                DatabaseBackend::Sqlite => {
                    let sqlite = SqliteJsonStore::connect(url).await?;
                    initialize_discovery_sqlite(&sqlite).await?;
                    if config.semantic_search.enabled {
                        semantic = SemanticRuntimeState::disabled(
                            "semantic_search_requires_postgres",
                            &config.semantic_search,
                        );
                    }
                    (Some(sqlite), None)
                }
                DatabaseBackend::Postgres => {
                    let postgres = PostgresJsonStore::connect(url).await?;
                    initialize_discovery_postgres(&postgres).await?;
                    semantic = initialize_semantic_runtime(&config, Some(&postgres)).await;
                    (None, Some(postgres))
                }
            }
        }
        _ => {
            if config.semantic_search.enabled {
                semantic = SemanticRuntimeState::disabled(
                    "semantic_search_requires_postgres",
                    &config.semantic_search,
                );
            }
            (None, None)
        }
    };
    let state = AppState {
        data: JsonStore::new(&config.paths.data_dir),
        index: JsonStore::new(&config.paths.index_dir),
        config: config.clone(),
        did: identity.did,
        sqlite,
        postgres,
        semantic,
        recommender: Arc::new(SemanticRecommender::new()?),
        client: reqwest::Client::new(),
        resource_sync_lock: Arc::new(Mutex::new(())),
        #[cfg(not(test))]
        late_package_retry_running: Arc::new(AtomicBool::new(false)),
        index_stats_cache: Arc::new(Mutex::new(None)),
    };
    Ok(state)
}

fn load_config(path: String) -> Result<Config> {
    let path = PathBuf::from(path);
    let mut config: Config = toml::from_str(&std::fs::read_to_string(&path)?)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    config.paths.data_dir = resolve_relative(base, &config.paths.data_dir);
    config.paths.index_dir = resolve_relative(base, &config.paths.index_dir);
    config.paths.identity_file = resolve_relative(base, &config.paths.identity_file);
    if let Some(database_url) = config.paths.database_url.as_mut() {
        *database_url = resolve_database_url(base, database_url);
    }
    Ok(config)
}

fn validate_stats_report_config(config: &StatsReportConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    if config
        .node_id
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(anyhow!("stats_report.node_id is required when enabled"));
    }
    if config
        .endpoint
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(anyhow!("stats_report.endpoint is required when enabled"));
    }
    if config
        .token
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(anyhow!("stats_report.token is required when enabled"));
    }
    if config.interval_seconds == 0 {
        return Err(anyhow!(
            "stats_report.interval_seconds must be greater than 0"
        ));
    }
    Ok(())
}

async fn initialize_discovery_sqlite(sqlite: &SqliteJsonStore) -> Result<()> {
    sqlite
        .execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {DISCOVERY_SYNC_STATE_TABLE} (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                sync_key TEXT UNIQUE,
                sync_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS {DISCOVERY_PACKAGE_TABLE} (
                resource_did TEXT PRIMARY KEY,
                cursor INTEGER NOT NULL,
                version TEXT NOT NULL,
                resource_type TEXT NOT NULL DEFAULT '',
                lifecycle_state TEXT NOT NULL DEFAULT '',
                capability_tags TEXT NOT NULL DEFAULT '[]',
                protocols TEXT NOT NULL DEFAULT '[]',
                service_endpoints TEXT NOT NULL DEFAULT '[]',
                tag_text TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                package_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS {DISCOVERY_REJECTED_TABLE} (
                reject_key TEXT PRIMARY KEY,
                item_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS {DISCOVERY_QUERY_STATS_TABLE} (
                stat_date TEXT PRIMARY KEY,
                query_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#
        ))
        .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "resource_type",
        "resource_type TEXT NOT NULL DEFAULT ''",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "lifecycle_state",
        "lifecycle_state TEXT NOT NULL DEFAULT ''",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "capability_tags",
        "capability_tags TEXT NOT NULL DEFAULT '[]'",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "protocols",
        "protocols TEXT NOT NULL DEFAULT '[]'",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "service_endpoints",
        "service_endpoints TEXT NOT NULL DEFAULT '[]'",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "tag_text",
        "tag_text TEXT NOT NULL DEFAULT ''",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_PACKAGE_TABLE,
        "search_text",
        "search_text TEXT NOT NULL DEFAULT ''",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_SYNC_STATE_TABLE,
        "state_key",
        "state_key TEXT",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_SYNC_STATE_TABLE,
        "state_value",
        "state_value TEXT",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_SYNC_STATE_TABLE,
        "sync_key",
        "sync_key TEXT",
    )
    .await?;
    sqlite_add_column_if_missing(
        sqlite,
        DISCOVERY_SYNC_STATE_TABLE,
        "sync_value",
        "sync_value TEXT",
    )
    .await?;
    sqlite
        .execute_batch(&format!(
            r#"
            UPDATE {DISCOVERY_SYNC_STATE_TABLE}
            SET state_key = COALESCE(state_key, sync_key),
                state_value = COALESCE(state_value, sync_value),
                sync_key = COALESCE(sync_key, state_key),
                sync_value = COALESCE(sync_value, state_value)
            WHERE state_key IS NULL
               OR state_value IS NULL
               OR sync_key IS NULL
               OR sync_value IS NULL;
            "#
        ))
        .await?;
    Ok(())
}

async fn sqlite_add_column_if_missing(
    sqlite: &SqliteJsonStore,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let escaped_table = table.replace('\'', "''");
    let rows = sqlx::query(&format!("PRAGMA table_info('{escaped_table}')"))
        .fetch_all(sqlite.pool())
        .await?;
    let exists = rows
        .iter()
        .any(|row| row.get::<String, _>("name") == column);
    if !exists {
        sqlx::query(&format!("ALTER TABLE {table} ADD COLUMN {definition}"))
            .execute(sqlite.pool())
            .await?;
    }
    Ok(())
}

fn resolve_relative(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn resolve_database_url(base: &Path, url: &str) -> String {
    let Some(raw_path) = url
        .strip_prefix("sqlite://")
        .or_else(|| url.strip_prefix("sqlite:"))
    else {
        return url.to_owned();
    };
    let resolved = resolve_relative(base, Path::new(raw_path));
    format!("sqlite:{}", resolved.display())
}

async fn initialize_discovery_postgres(postgres: &PostgresJsonStore) -> Result<()> {
    postgres
        .execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {DISCOVERY_SYNC_STATE_TABLE} (
                state_key TEXT PRIMARY KEY,
                state_value JSONB NOT NULL,
                sync_key TEXT UNIQUE,
                sync_value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS {DISCOVERY_PACKAGE_TABLE} (
                resource_did TEXT PRIMARY KEY,
                cursor BIGINT NOT NULL,
                version TEXT NOT NULL,
                resource_type TEXT NOT NULL DEFAULT '',
                lifecycle_state TEXT NOT NULL DEFAULT '',
                capability_tags TEXT[] NOT NULL DEFAULT '{{}}',
                protocols TEXT[] NOT NULL DEFAULT '{{}}',
                service_endpoints TEXT[] NOT NULL DEFAULT '{{}}',
                tag_text TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                package_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS resource_type TEXT NOT NULL DEFAULT '';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS lifecycle_state TEXT NOT NULL DEFAULT '';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS capability_tags TEXT[] NOT NULL DEFAULT '{{}}';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS protocols TEXT[] NOT NULL DEFAULT '{{}}';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS service_endpoints TEXT[] NOT NULL DEFAULT '{{}}';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS tag_text TEXT NOT NULL DEFAULT '';
            ALTER TABLE {DISCOVERY_PACKAGE_TABLE}
                ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT '';
            CREATE TABLE IF NOT EXISTS {DISCOVERY_REJECTED_TABLE} (
                reject_key TEXT PRIMARY KEY,
                item_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS {DISCOVERY_QUERY_STATS_TABLE} (
                stat_date TEXT PRIMARY KEY,
                query_count BIGINT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_cursor
            ON {DISCOVERY_PACKAGE_TABLE}(cursor DESC);
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_updated
            ON {DISCOVERY_PACKAGE_TABLE}(updated_at, resource_did);
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_type_state
            ON {DISCOVERY_PACKAGE_TABLE}(resource_type, lifecycle_state);
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_capability_tags
            ON {DISCOVERY_PACKAGE_TABLE} USING GIN(capability_tags);
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_protocols
            ON {DISCOVERY_PACKAGE_TABLE} USING GIN(protocols);
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_tag_text
            ON {DISCOVERY_PACKAGE_TABLE} USING GIN(to_tsvector('simple', tag_text));
            CREATE INDEX IF NOT EXISTS idx_discovery_packages_search_text
            ON {DISCOVERY_PACKAGE_TABLE} USING GIN(to_tsvector('simple', search_text));
            CREATE INDEX IF NOT EXISTS idx_discovery_rejected_updated
            ON {DISCOVERY_REJECTED_TABLE}(updated_at, reject_key);
            "#
        ))
        .await?;
    postgres
        .execute_batch(&format!(
            r#"
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE} ADD COLUMN IF NOT EXISTS state_key TEXT;
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE} ADD COLUMN IF NOT EXISTS state_value JSONB;
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE} ADD COLUMN IF NOT EXISTS sync_key TEXT;
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE} ADD COLUMN IF NOT EXISTS sync_value JSONB;
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE}
                ALTER COLUMN state_value TYPE JSONB
                USING state_value::jsonb;
            ALTER TABLE {DISCOVERY_SYNC_STATE_TABLE}
                ALTER COLUMN sync_value TYPE JSONB
                USING sync_value::jsonb;
            UPDATE {DISCOVERY_SYNC_STATE_TABLE}
            SET state_key = COALESCE(state_key, sync_key),
                state_value = COALESCE(state_value, sync_value),
                sync_key = COALESCE(sync_key, state_key),
                sync_value = COALESCE(sync_value, state_value)
            WHERE state_key IS NULL
               OR state_value IS NULL
               OR sync_key IS NULL
               OR sync_value IS NULL;
            "#
        ))
        .await?;
    Ok(())
}

async fn initialize_semantic_runtime(
    config: &Config,
    postgres: Option<&PostgresJsonStore>,
) -> SemanticRuntimeState {
    if !config.semantic_search.enabled {
        return SemanticRuntimeState::disabled("semantic_disabled", &config.semantic_search);
    }
    let engine = match DiscoverySearchEngine::from_config(&config.semantic_search) {
        Ok(engine) => engine,
        Err(err) => {
            return SemanticRuntimeState::disabled(err.to_string(), &config.semantic_search);
        }
    };
    if let Err(err) =
        validate_embedding_provider(&config.semantic_search, &reqwest::Client::new()).await
    {
        return SemanticRuntimeState::disabled(err.to_string(), &config.semantic_search);
    }
    let Some(postgres) = postgres else {
        return SemanticRuntimeState::disabled(
            "semantic_search_requires_postgres",
            &config.semantic_search,
        );
    };
    match engine.initialize(postgres, &config.semantic_search).await {
        Ok(()) => SemanticRuntimeState {
            enabled: true,
            healthy: true,
            backend: config.semantic_search.engine.clone(),
            fallback_reason: None,
            embedding_model: config.semantic_search.embedding.model_name.clone(),
            embedding_version: semantic_embedding_version(&config.semantic_search),
            embedding_dtype: config.semantic_search.embedding.dtype.clone(),
            dimension: config.semantic_search.embedding.dimension,
        },
        Err(err) => {
            eprintln!("semantic search disabled: {err}");
            SemanticRuntimeState::disabled(
                format!("semantic_initialization_failed:{err}"),
                &config.semantic_search,
            )
        }
    }
}

async fn initialize_semantic_postgres(
    postgres: &PostgresJsonStore,
    config: &SemanticSearchConfig,
) -> Result<()> {
    let dimension = config.embedding.dimension;
    if dimension == 0 || dimension > 4096 {
        return Err(anyhow!("invalid_semantic_embedding_dimension"));
    }
    postgres
        .execute_batch("CREATE EXTENSION IF NOT EXISTS vector;")
        .await?;
    postgres
        .execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS {DISCOVERY_SEMANTIC_INDEX_TABLE} (
                resource_did TEXT PRIMARY KEY,
                cursor BIGINT NOT NULL,
                package_version TEXT NOT NULL,
                did_document_hash TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                package_hash TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                capability_tags TEXT[] NOT NULL DEFAULT '{{}}',
                protocols TEXT[] NOT NULL DEFAULT '{{}}',
                service_endpoints TEXT[] NOT NULL DEFAULT '{{}}',
                name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                examples JSONB NOT NULL DEFAULT '[]'::jsonb,
                use_cases JSONB NOT NULL DEFAULT '[]'::jsonb,
                tag_text TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                semantic_source_hash TEXT NOT NULL,
                embedding VECTOR({dimension}),
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL DEFAULT '1',
                updated_at TIMESTAMPTZ NOT NULL
            );
            ALTER TABLE {DISCOVERY_SEMANTIC_INDEX_TABLE}
                ADD COLUMN IF NOT EXISTS tag_text TEXT NOT NULL DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_cursor
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE}(cursor DESC);
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_type_state
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE}(resource_type, lifecycle_state);
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_tags
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE} USING GIN(capability_tags);
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_protocols
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE} USING GIN(protocols);
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_tag_text
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE} USING GIN(to_tsvector('simple', tag_text));
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_search_text
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE} USING GIN(to_tsvector('simple', search_text));
            CREATE INDEX IF NOT EXISTS idx_discovery_semantic_embedding
            ON {DISCOVERY_SEMANTIC_INDEX_TABLE}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);

            CREATE TABLE IF NOT EXISTS {DISCOVERY_INTENT_INDEX_TABLE} (
                resource_did TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                intent_kind TEXT NOT NULL,
                intent_text TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                semantic_source_hash TEXT NOT NULL,
                embedding VECTOR({dimension}),
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(resource_did, intent_id, embedding_model, embedding_version)
            );
            CREATE INDEX IF NOT EXISTS idx_discovery_intent_resource
            ON {DISCOVERY_INTENT_INDEX_TABLE}(resource_did);
            CREATE INDEX IF NOT EXISTS idx_discovery_intent_type_state
            ON {DISCOVERY_INTENT_INDEX_TABLE}(resource_type, lifecycle_state);
            CREATE INDEX IF NOT EXISTS idx_discovery_intent_embedding
            ON {DISCOVERY_INTENT_INDEX_TABLE}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            "#,
        ))
        .await?;
    Ok(())
}

fn semantic_embedding_version(config: &SemanticSearchConfig) -> String {
    config.embedding.model_version.clone().unwrap_or_else(|| {
        format!(
            "1:{}:{}",
            config.embedding.provider, config.embedding.dimension
        )
    })
}

fn semantic_search_available(state: &AppState) -> bool {
    state.config.semantic_search.enabled
        && state.semantic.enabled
        && state.semantic.healthy
        && state.postgres.is_some()
}

fn semantic_diagnostics(
    state: &AppState,
    fallback_used: bool,
    fallback_reason: Option<String>,
) -> SemanticSearchDiagnostics {
    SemanticSearchDiagnostics {
        enabled: state.semantic.enabled,
        healthy: state.semantic.healthy,
        backend: state.semantic.backend.clone(),
        embedding_model: state.semantic.embedding_model.clone(),
        embedding_version: state.semantic.embedding_version.clone(),
        embedding_dtype: state.semantic.embedding_dtype.clone(),
        dimension: state.semantic.dimension,
        fallback_used,
        fallback_reason: fallback_reason.or_else(|| state.semantic.fallback_reason.clone()),
    }
}

fn semantic_status_json(
    state: &AppState,
    fallback_used: bool,
    fallback_reason: Option<String>,
) -> Value {
    let diagnostics = semantic_diagnostics(state, fallback_used, fallback_reason);
    json!({
        "enabled": diagnostics.enabled,
        "healthy": diagnostics.healthy,
        "backend": diagnostics.backend,
        "fallback": state.config.semantic_search.fallback,
        "defaultMode": state.config.semantic_search.default_mode,
        "explainEnabled": state.config.semantic_search.explain_enabled,
        "embeddingProvider": state.config.semantic_search.embedding.provider,
        "embeddingModel": diagnostics.embedding_model,
        "embeddingVersion": diagnostics.embedding_version,
        "embeddingDtype": diagnostics.embedding_dtype,
        "modelPath": state.config.semantic_search.embedding.model_path,
        "dimension": diagnostics.dimension,
        "timeoutMs": state.config.semantic_search.embedding.timeout_ms,
        "batchSize": state.config.semantic_search.embedding.batch_size,
        "filters": {
            "resourceType": state.config.semantic_search.filters.resource_type,
            "capabilityTags": state.config.semantic_search.filters.capability_tags,
            "protocol": state.config.semantic_search.filters.protocol
        },
        "fallbackUsed": diagnostics.fallback_used,
        "fallbackReason": diagnostics.fallback_reason,
    })
}

fn normalize_semantic_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

fn push_unique_text(values: &mut Vec<String>, value: impl AsRef<str>) {
    let normalized = normalize_semantic_text(value.as_ref());
    if !normalized.is_empty() && !values.iter().any(|existing| existing == &normalized) {
        values.push(normalized);
    }
}

fn json_example_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            let normalized = normalize_semantic_text(text);
            (!normalized.is_empty()).then_some(normalized)
        }
        Value::Object(_) | Value::Array(_) => {
            let text = value.to_string();
            let normalized = normalize_semantic_text(&text);
            (!normalized.is_empty()).then_some(normalized.chars().take(1_000).collect())
        }
        _ => None,
    }
}

fn search_document_from_package(
    cursor: i64,
    package: &ResourcePackage,
    projection: &DiscoveryPackageProjection,
) -> SearchDocument {
    let mut descriptions = Vec::new();
    push_unique_text(&mut descriptions, &package.metadata.description);
    let mut examples = Vec::new();
    let mut use_cases = Vec::new();
    let mut tags = package.metadata.capability_tags.clone();

    if let Some(metadata) = &package.did_document.oan_metadata {
        tags.extend(metadata.capability_tags.iter().cloned());
        if let Some(description) = &metadata.resource_description {
            if let Some(name) = &description.name {
                push_unique_text(&mut descriptions, name);
            }
            if let Some(text) = &description.description {
                push_unique_text(&mut descriptions, text);
            }
            if let Some(text) = &description.capability_description {
                push_unique_text(&mut descriptions, text);
            }
            tags.extend(description.capability_tags.iter().cloned());
            for item in description.use_case_examples.iter().take(20) {
                push_unique_text(&mut use_cases, item);
            }
            for value in description.examples.iter().take(20) {
                if let Some(text) = json_example_to_text(value) {
                    push_unique_text(&mut examples, text);
                }
            }
        }
        if let Some(description) = &metadata.agent_description {
            push_unique_text(&mut descriptions, &description.capability_description);
            tags.extend(description.capability_tags.iter().cloned());
            for item in description.use_case_examples.iter().take(20) {
                push_unique_text(&mut use_cases, item);
            }
        }
    }

    tags.sort();
    tags.dedup();
    let description = descriptions.join("\n");
    let name = package
        .did_document
        .oan_metadata
        .as_ref()
        .and_then(|metadata| metadata.resource_description.as_ref())
        .and_then(|description| description.name.clone())
        .unwrap_or_else(|| package.metadata.name.clone());
    let embedding_text = semantic_embedding_text(&SemanticEmbeddingTextInput {
        name: &name,
        resource_type: &projection.resource_type,
        description: &description,
        capability_tags: &tags,
        protocols: &projection.protocols,
        service_endpoints: &projection.service_endpoints,
        examples: &examples,
        use_cases: &use_cases,
    });
    SearchDocument {
        resource_did: package.resource_did.clone(),
        cursor,
        package_version: package.package_version.clone(),
        did_document_hash: package.did_document_hash.clone(),
        metadata_hash: package.metadata_hash.clone(),
        package_hash: package.package_hash.clone(),
        resource_type: projection.resource_type.clone(),
        lifecycle_state: projection.lifecycle_state.clone(),
        capability_tags: tags,
        protocols: projection.protocols.clone(),
        service_endpoints: projection.service_endpoints.clone(),
        name,
        description,
        examples,
        use_cases,
        tag_text: projection.tag_text.clone(),
        search_text: projection.search_text.clone(),
        semantic_source_hash: format!("sha256:{}", sha256_hex(embedding_text.as_bytes())),
    }
}

fn semantic_search_text(values: &[String]) -> String {
    let raw_text = values
        .iter()
        .map(|value| normalize_semantic_text(value))
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let tokens = semantic_tokens(&raw_text);
    normalize_semantic_text(&format!("{} {}", raw_text.to_lowercase(), tokens.join(" ")))
}

struct SemanticEmbeddingTextInput<'a> {
    name: &'a str,
    resource_type: &'a str,
    description: &'a str,
    capability_tags: &'a [String],
    protocols: &'a [String],
    service_endpoints: &'a [String],
    examples: &'a [String],
    use_cases: &'a [String],
}

fn semantic_embedding_text(input: &SemanticEmbeddingTextInput<'_>) -> String {
    let mut text = String::new();
    text.push_str("Name:\n");
    text.push_str(input.name);
    text.push_str("\n\nResource type:\n");
    text.push_str(input.resource_type);
    text.push_str("\n\nDescription:\n");
    text.push_str(input.description);
    text.push_str("\n\nCapability tags:\n");
    text.push_str(&input.capability_tags.join(", "));
    text.push_str("\n\nProtocols:\n");
    text.push_str(&input.protocols.join(", "));
    text.push_str("\n\nServices:\n");
    text.push_str(&input.service_endpoints.join("\n"));
    text.push_str("\n\nExamples:\n");
    text.push_str(&input.examples.join("\n"));
    text.push_str("\n\nUse cases:\n");
    text.push_str(&input.use_cases.join("\n"));
    text.push_str("\n\nTag view:\n");
    for tag in input.capability_tags {
        text.push_str("capability ");
        text.push_str(tag);
        text.push('\n');
    }
    text.push_str("\nContext view:\n");
    text.push_str(input.name);
    text.push_str(" provides ");
    text.push_str(input.resource_type);
    text.push_str(" capabilities for ");
    text.push_str(input.description);
    text.push_str("\n\nIntent view:\n");
    for use_case in input.use_cases {
        text.push_str("User wants to ");
        text.push_str(use_case);
        text.push('\n');
    }
    for example in input.examples {
        text.push_str("Example request: ");
        text.push_str(example);
        text.push('\n');
    }
    text.chars().take(8_000).collect()
}

fn semantic_document_text(doc: &SearchDocument) -> String {
    semantic_embedding_text(&SemanticEmbeddingTextInput {
        name: &doc.name,
        resource_type: &doc.resource_type,
        description: &doc.description,
        capability_tags: &doc.capability_tags,
        protocols: &doc.protocols,
        service_endpoints: &doc.service_endpoints,
        examples: &doc.examples,
        use_cases: &doc.use_cases,
    })
}

fn semantic_alias_catalog() -> &'static SemanticAliasCatalog {
    static CATALOG: OnceLock<SemanticAliasCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let alias_file: SemanticAliasFile =
            serde_json::from_str(SEMANTIC_ALIASES_JSON).expect("semantic_aliases.v1.json is valid");
        let mut phrase_to_canonical = Vec::new();
        let mut canonical_to_terms = BTreeMap::new();
        for group in alias_file.groups {
            let canonical = normalize_semantic_token(&group.canonical);
            let mut terms = Vec::new();
            push_unique_text(&mut terms, &canonical);
            for term in group.terms {
                push_unique_text(&mut terms, term);
            }
            for term in &terms {
                phrase_to_canonical.push((term.to_lowercase(), canonical.clone()));
            }
            canonical_to_terms.insert(canonical, terms);
        }
        phrase_to_canonical.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()));
        SemanticAliasCatalog {
            phrase_to_canonical,
            canonical_to_terms,
        }
    })
}

fn semantic_jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

fn normalize_semantic_token(token: &str) -> String {
    token.trim().to_lowercase()
}

fn technical_semantic_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| {
        !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | ':' | '_' | '-' | '/' | '#'))
    })
    .filter_map(|token| {
        let token = normalize_semantic_token(token)
            .trim_matches(|ch: char| matches!(ch, '.' | ':' | '_' | '-' | '/' | '#'))
            .to_owned();
        (token.len() >= 2).then_some(token)
    })
    .collect()
}

fn expand_semantic_aliases(tokens: &[String], raw_text: &str) -> Vec<String> {
    let catalog = semantic_alias_catalog();
    let lower_text = raw_text.to_lowercase();
    let mut expanded = tokens.to_vec();
    let mut seen = expanded.iter().cloned().collect::<HashSet<_>>();
    for (phrase, canonical) in &catalog.phrase_to_canonical {
        if lower_text.contains(phrase) || tokens.iter().any(|token| token == phrase) {
            if seen.insert(canonical.clone()) {
                expanded.push(canonical.clone());
            }
            if let Some(terms) = catalog.canonical_to_terms.get(canonical) {
                for term in terms.iter().take(8) {
                    for token in semantic_tokens_without_alias_expansion(term) {
                        if seen.insert(token.clone()) {
                            expanded.push(token);
                        }
                    }
                }
            }
        }
    }
    expanded
}

fn semantic_tokens_without_alias_expansion(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();
    for token in technical_semantic_tokens(text) {
        if seen.insert(token.clone()) {
            tokens.push(token);
        }
    }
    for token in semantic_jieba().cut(text, false) {
        let normalized = normalize_semantic_token(token);
        if normalized.chars().count() >= 2 && seen.insert(normalized.clone()) {
            tokens.push(normalized);
        }
    }
    tokens
}

fn deterministic_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let dimension = dimension.max(1);
    let mut vector = vec![0.0_f32; dimension];
    let mut seen = HashSet::new();
    for token in semantic_tokens(text) {
        if !seen.insert(token.clone()) {
            continue;
        }
        let hash = sha256_hex(token.as_bytes());
        let index = usize::from_str_radix(&hash[0..8], 16).unwrap_or(0) % dimension;
        vector[index] += 1.0;
    }
    normalize_vector(vector)
}

async fn validate_embedding_provider(
    config: &SemanticSearchConfig,
    client: &reqwest::Client,
) -> Result<()> {
    match EmbeddingProviderKind::from_config(&config.embedding)? {
        EmbeddingProviderKind::DeterministicLocal => {
            let probe = deterministic_embedding("oan semantic probe", config.embedding.dimension);
            validate_embedding_vector(&probe, config.embedding.dimension)?;
            Ok(())
        }
        EmbeddingProviderKind::HttpEmbedding => {
            let health_endpoint = config
                .embedding
                .health_endpoint
                .as_deref()
                .ok_or_else(|| anyhow!("http_embedding_health_endpoint_required"))?;
            let health = client
                .get(health_endpoint)
                .timeout(StdDuration::from_millis(config.embedding.timeout_ms))
                .send()
                .await?
                .error_for_status()?
                .json::<HttpEmbeddingHealthResponse>()
                .await?;
            if !health.ready {
                return Err(anyhow!("http_embedding_provider_not_ready"));
            }
            if health.model != config.embedding.model_name {
                return Err(anyhow!("http_embedding_model_mismatch:{}", health.model));
            }
            if health.dimension != config.embedding.dimension {
                return Err(anyhow!(
                    "http_embedding_dimension_mismatch:{}",
                    health.dimension
                ));
            }
            if config.embedding.strict_model_version {
                let expected = config
                    .embedding
                    .model_version
                    .as_deref()
                    .ok_or_else(|| anyhow!("strict_model_version_requires_modelVersion"))?;
                let actual = health
                    .embedding_version
                    .as_deref()
                    .ok_or_else(|| anyhow!("http_embedding_version_missing"))?;
                if actual != expected {
                    return Err(anyhow!("http_embedding_version_mismatch:{actual}"));
                }
            }
            if let Some(expected) = config.embedding.dtype.as_deref() {
                let actual = health
                    .dtype
                    .as_deref()
                    .ok_or_else(|| anyhow!("http_embedding_dtype_missing"))?;
                if actual != expected {
                    return Err(anyhow!("http_embedding_dtype_mismatch:{actual}"));
                }
            }
            let probe = embed_semantic_text(config, client, "oan semantic probe").await?;
            validate_embedding_vector(&probe, config.embedding.dimension)
        }
    }
}

async fn embed_semantic_text(
    config: &SemanticSearchConfig,
    client: &reqwest::Client,
    text: &str,
) -> Result<Vec<f32>> {
    let text = bounded_embedding_text(text, config.embedding.max_input_chars);
    match EmbeddingProviderKind::from_config(&config.embedding)? {
        EmbeddingProviderKind::DeterministicLocal => {
            Ok(deterministic_embedding(&text, config.embedding.dimension))
        }
        EmbeddingProviderKind::HttpEmbedding => {
            let endpoint = config
                .embedding
                .endpoint
                .as_deref()
                .ok_or_else(|| anyhow!("http_embedding_endpoint_required"))?;
            let response = client
                .post(endpoint)
                .timeout(StdDuration::from_millis(config.embedding.timeout_ms))
                .json(&HttpEmbeddingRequest {
                    model: config.embedding.model_name.clone(),
                    input: vec![text],
                })
                .send()
                .await?
                .error_for_status()?
                .json::<HttpEmbeddingResponse>()
                .await?;
            if response.model != config.embedding.model_name {
                return Err(anyhow!("http_embedding_model_mismatch:{}", response.model));
            }
            if response.dimension != config.embedding.dimension {
                return Err(anyhow!(
                    "http_embedding_dimension_mismatch:{}",
                    response.dimension
                ));
            }
            if config.embedding.strict_model_version {
                let expected = config
                    .embedding
                    .model_version
                    .as_deref()
                    .ok_or_else(|| anyhow!("strict_model_version_requires_modelVersion"))?;
                if response.embedding_version != expected {
                    return Err(anyhow!(
                        "http_embedding_version_mismatch:{}",
                        response.embedding_version
                    ));
                }
            }
            if let Some(expected) = config.embedding.dtype.as_deref() {
                let actual = response
                    .dtype
                    .as_deref()
                    .ok_or_else(|| anyhow!("http_embedding_response_dtype_missing"))?;
                if actual != expected {
                    return Err(anyhow!("http_embedding_response_dtype_mismatch:{actual}"));
                }
            }
            let vector = response
                .vectors
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("http_embedding_vector_missing"))?;
            validate_embedding_vector(&vector, config.embedding.dimension)?;
            Ok(normalize_vector(vector))
        }
    }
}

fn bounded_embedding_text(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars.max(1)).collect()
}

fn validate_embedding_vector(vector: &[f32], dimension: usize) -> Result<()> {
    if vector.len() != dimension {
        return Err(anyhow!(
            "embedding_dimension_mismatch:{}:{}",
            vector.len(),
            dimension
        ));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("embedding_vector_contains_non_finite_value"));
    }
    Ok(())
}

fn semantic_tokens(text: &str) -> Vec<String> {
    let tokens = semantic_tokens_without_alias_expansion(text);
    expand_semantic_aliases(&tokens, text)
}

fn semantic_query_projection(query: &ResourceDiscoveryQuery) -> SemanticQueryProjection {
    let raw_text = query.query.as_deref().unwrap_or_default().trim().to_owned();
    let mut explicit_terms = HashSet::new();
    if let Some(resource_type) = &query.resource_type {
        explicit_terms.insert(resource_type.as_str().to_owned());
    }
    if let Some(protocol) = &query.protocol {
        explicit_terms.insert(protocol.to_ascii_lowercase());
    }
    explicit_terms.extend(
        query
            .capability_tags
            .iter()
            .map(|tag| tag.to_ascii_lowercase()),
    );

    let tokens = semantic_tokens_without_alias_expansion(&raw_text);
    let mut expanded_tokens = expand_semantic_aliases(&tokens, &raw_text);
    for term in &explicit_terms {
        for token in semantic_tokens(term) {
            if !expanded_tokens.iter().any(|existing| existing == &token) {
                expanded_tokens.push(token);
            }
        }
    }
    SemanticQueryProjection { expanded_tokens }
}

fn semantic_query_tag_tokens(query: &ResourceDiscoveryQuery) -> Vec<String> {
    let mut tokens = Vec::new();
    if let Some(resource_type) = &query.resource_type {
        tokens.extend(semantic_tokens(resource_type.as_str()));
    }
    if let Some(protocol) = &query.protocol {
        tokens.extend(semantic_tokens(protocol));
    }
    for tag in &query.capability_tags {
        tokens.extend(semantic_tokens(tag));
    }
    let query_text = query.query.as_deref().unwrap_or_default();
    if !query_text.trim().is_empty() {
        tokens.extend(semantic_tokens(query_text));
    }
    let mut seen = HashSet::new();
    tokens
        .into_iter()
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

fn semantic_tag_text(doc: &SearchDocument) -> String {
    [
        doc.resource_did.as_str(),
        doc.resource_type.as_str(),
        &doc.capability_tags.join(" "),
        &doc.protocols.join(" "),
        &doc.service_endpoints.join(" "),
        doc.tag_text.as_str(),
    ]
    .join(" ")
}

fn semantic_context_text(doc: &SearchDocument) -> String {
    [
        doc.name.as_str(),
        doc.resource_type.as_str(),
        doc.description.as_str(),
        &doc.capability_tags.join(" "),
        &doc.protocols.join(" "),
        &doc.use_cases.join(" "),
        &doc.examples.join(" "),
        doc.search_text.as_str(),
    ]
    .join(" ")
}

fn semantic_lexical_text(doc: &SearchDocument) -> String {
    let tag_text = semantic_tag_text(doc);
    let context_text = semantic_context_text(doc);
    format!("{tag_text} {tag_text} {context_text}")
}

fn semantic_intent_texts(doc: &SearchDocument) -> Vec<String> {
    let mut texts = Vec::new();
    for value in &doc.use_cases {
        push_unique_text(&mut texts, value);
    }
    for value in &doc.examples {
        push_unique_text(&mut texts, value);
    }
    if !doc.description.is_empty() {
        push_unique_text(
            &mut texts,
            format!("Find a {} for {}", doc.resource_type, doc.description),
        );
        push_unique_text(
            &mut texts,
            format!("Use {} to {}", doc.name, doc.description),
        );
    }
    for tag in &doc.capability_tags {
        push_unique_text(&mut texts, format!("Find a resource for capability {tag}"));
    }
    texts.into_iter().take(16).collect()
}

fn semantic_intent_documents(doc: &SearchDocument) -> Vec<IntentDocument> {
    semantic_intent_texts(doc)
        .into_iter()
        .enumerate()
        .map(|(index, intent_text)| {
            let intent_kind = if doc.use_cases.iter().any(|value| value == &intent_text) {
                "use_case"
            } else if doc.examples.iter().any(|value| value == &intent_text) {
                "example"
            } else {
                "pseudo_query"
            };
            let intent_id = format!("intent-{:02}", index + 1);
            IntentDocument {
                resource_did: doc.resource_did.clone(),
                intent_id,
                intent_kind: intent_kind.to_owned(),
                semantic_source_hash: format!("sha256:{}", sha256_hex(intent_text.as_bytes())),
                intent_text,
                resource_type: doc.resource_type.clone(),
                lifecycle_state: doc.lifecycle_state.clone(),
            }
        })
        .collect()
}

fn token_counts(tokens: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }
    counts
}

fn weighted_token_similarity(query_tokens: &[String], text: &str) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let doc_tokens = semantic_tokens(text);
    if doc_tokens.is_empty() {
        return 0.0;
    }
    let doc_counts = token_counts(&doc_tokens);
    let matched = query_tokens
        .iter()
        .filter(|token| doc_counts.contains_key(*token))
        .count();
    (matched as f32 / query_tokens.len() as f32).clamp(0.0, 1.0)
}

fn bm25_like_score(query_tokens: &[String], documents: &[Vec<String>], index: usize) -> f32 {
    if query_tokens.is_empty() || documents.is_empty() || index >= documents.len() {
        return 0.0;
    }
    let doc_tokens = &documents[index];
    if doc_tokens.is_empty() {
        return 0.0;
    }
    let doc_counts = token_counts(doc_tokens);
    let avg_len = documents
        .iter()
        .map(|tokens| tokens.len() as f32)
        .sum::<f32>()
        / documents.len() as f32;
    let doc_len = doc_tokens.len() as f32;
    let k1 = 1.2_f32;
    let b = 0.75_f32;
    let mut score = 0.0;
    let mut unique_query_terms = HashSet::new();
    for token in query_tokens {
        if !unique_query_terms.insert(token) {
            continue;
        }
        let Some(tf) = doc_counts.get(token).copied() else {
            continue;
        };
        let doc_freq = documents
            .iter()
            .filter(|tokens| tokens.contains(token))
            .count() as f32;
        let idf = (((documents.len() as f32 - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0).ln();
        let tf = tf as f32;
        score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avg_len.max(1.0)));
    }
    (score / (score + 3.0)).clamp(0.0, 1.0)
}

fn semantic_resource_scores(
    package: &ResourcePackage,
    query: &ResourceDiscoveryQuery,
    query_projection: &SemanticQueryProjection,
    bm25_score: f32,
) -> SemanticQueryHit {
    let projection = discovery_package_projection(package);
    let doc = search_document_from_package(0, package, &projection);
    let tag_score =
        weighted_token_similarity(&query_projection.expanded_tokens, &semantic_tag_text(&doc));
    let context_score = weighted_token_similarity(
        &query_projection.expanded_tokens,
        &semantic_context_text(&doc),
    );
    let (intent_score, matched_intent) = semantic_intent_texts(&doc)
        .into_iter()
        .map(|intent| {
            let score = weighted_token_similarity(&query_projection.expanded_tokens, &intent);
            (score, intent)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .filter(|(score, _)| *score > 0.0)
        .map(|(score, intent)| (score, Some(intent)))
        .unwrap_or((0.0, None));
    let structured_score = semantic_structured_score(package, query);
    let (tag_weight, context_weight, tag_bonus) = if query.resource_type.is_some() {
        (0.25, 0.25, 0.0)
    } else {
        let tag_bonus = if tag_score > 0.0 { 0.04 } else { 0.0 };
        (0.34, 0.16, tag_bonus)
    };
    let semantic_score = (0.30 * bm25_score)
        + (tag_weight * tag_score)
        + (context_weight * context_score)
        + (0.20 * intent_score);
    let final_score = (semantic_score + structured_score + tag_bonus).clamp(0.0, 1.0);
    SemanticQueryHit {
        package: package.clone(),
        semantic_score,
        bm25_score,
        tag_score,
        context_score,
        intent_score,
        structured_score,
        final_score,
        matched_intent,
    }
}

fn text_has_cjk(text: &str) -> bool {
    text.chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
}

fn should_use_sql_text_prefilter(query: &ResourceDiscoveryQuery) -> bool {
    query
        .query
        .as_deref()
        .map(|text| {
            let trimmed = text.trim();
            !trimmed.is_empty() && !text_has_cjk(trimmed)
        })
        .unwrap_or(false)
}

fn rank_resource_packages_semantically(
    packages: Vec<ResourcePackage>,
    query: &ResourceDiscoveryQuery,
) -> Vec<SemanticQueryHit> {
    let query_projection = semantic_query_projection(query);
    let doc_tokens = packages
        .iter()
        .map(|package| {
            let projection = discovery_package_projection(package);
            let doc = search_document_from_package(0, package, &projection);
            semantic_tokens(&semantic_lexical_text(&doc))
        })
        .collect::<Vec<_>>();
    let has_semantic_terms = !query_projection.expanded_tokens.is_empty();
    let mut hits = packages
        .iter()
        .enumerate()
        .filter(|(_, package)| resource_matches_explicit_constraints(package, query))
        .map(|(index, package)| {
            let bm25_score = bm25_like_score(&query_projection.expanded_tokens, &doc_tokens, index);
            semantic_resource_scores(package, query, &query_projection, bm25_score)
        })
        .filter(|hit| !has_semantic_terms || hit.final_score > 0.0)
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.final_score
            .total_cmp(&a.final_score)
            .then_with(|| {
                b.package
                    .metadata
                    .updated_at
                    .cmp(&a.package.metadata.updated_at)
            })
            .then_with(|| a.package.resource_did.cmp(&b.package.resource_did))
    });
    hits
}

fn semantic_match_reason(hit: &SemanticQueryHit) -> String {
    let mut reasons = Vec::new();
    if hit.bm25_score > 0.0 {
        reasons.push("lexical terms");
    }
    if hit.tag_score > 0.0 {
        reasons.push("capability tags or protocols");
    }
    if hit.context_score > 0.0 {
        reasons.push("resource context");
    }
    if hit.intent_score > 0.0 {
        reasons.push("examples or use cases");
    }
    if hit.structured_score > 0.0 {
        reasons.push("explicit filters");
    }
    if reasons.is_empty() {
        "No semantic match above zero; returned only by structured eligibility.".to_owned()
    } else {
        format!("Matched by {}.", reasons.join(", "))
    }
}

fn normalize_vector(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn pgvector_literal(vector: &[f32]) -> String {
    format!(
        "'[{}]'",
        vector
            .iter()
            .map(|value| format!("{value:.8}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn semantic_structured_score(package: &ResourcePackage, query: &ResourceDiscoveryQuery) -> f32 {
    let mut score = 0.0;
    if query
        .resource_type
        .as_ref()
        .is_some_and(|resource_type| resource_type == &package.resource_type)
    {
        score += 0.10;
    }
    if !query.capability_tags.is_empty() {
        let overlap = query
            .capability_tags
            .iter()
            .filter(|tag| package.metadata.capability_tags.contains(tag))
            .count();
        score += (overlap as f32 / query.capability_tags.len() as f32) * 0.15;
    }
    if query.protocol.as_ref().is_some_and(|protocol| {
        package.metadata.services.iter().any(|service| {
            service
                .protocol
                .as_ref()
                .map(|candidate| candidate.eq_ignore_ascii_case(protocol))
                .unwrap_or(false)
        }) || package.metadata.protocol_bindings.iter().any(|binding| {
            binding["protocol"]
                .as_str()
                .map(|candidate| candidate.eq_ignore_ascii_case(protocol))
                .unwrap_or(false)
        })
    }) {
        score += 0.05;
    }
    score
}

fn is_user_consumable_resource_type(resource_type: &ResourceType) -> bool {
    matches!(
        resource_type,
        ResourceType::AgentService
            | ResourceType::Skill
            | ResourceType::McpServer
            | ResourceType::ToolApi
    )
}

fn should_search_resource_type(package: &ResourcePackage, query: &ResourceDiscoveryQuery) -> bool {
    query.resource_type.is_some() || is_user_consumable_resource_type(&package.resource_type)
}

fn default_discoverable_resource_type_names() -> Vec<String> {
    ["agent_service", "skill", "mcp_server", "tool_api"]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect()
}

fn build_cors_layer(config: &CorsConfig) -> Result<CorsLayer> {
    let origins: Vec<HeaderValue> = config
        .allowed_origins
        .iter()
        .map(|origin| HeaderValue::from_str(origin))
        .collect::<std::result::Result<_, _>>()?;
    Ok(CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::OPTIONS])
        .allow_headers(AllowHeaders::any()))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_owned(),
        node_type: "discovery".to_owned(),
        did: Some(state.did),
    })
}

async fn discovery_did_document(State(state): State<AppState>) -> ApiResult<DidDocument> {
    state
        .data
        .read("did-document.json")
        .map(Json)
        .map_err(|err| ApiError::internal(err.into()))
}

async fn discovery_debug_export_loop(state: AppState) {
    loop {
        if let Err(err) = export_discovery_debug_snapshot(&state).await {
            eprintln!("discovery debug export failed: {err}");
        }
        sleep(TokioDuration::from_millis(
            state.config.debug.export_interval_ms.max(100),
        ))
        .await;
    }
}

#[cfg(not(test))]
fn wake_late_package_retry_task(state: AppState) {
    let preflight = state.clone();
    if state
        .late_package_retry_running
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    tokio::spawn(async move {
        if !has_late_resource_packages(&preflight).await {
            preflight
                .late_package_retry_running
                .store(false, Ordering::SeqCst);
            if has_late_resource_packages(&preflight).await {
                wake_late_package_retry_task(preflight);
            }
            return;
        }
        late_package_retry_loop(state).await;
    });
}

#[cfg(not(test))]
async fn has_late_resource_packages(state: &AppState) -> bool {
    count_late_rejected_packages(state)
        .await
        .map(|count| count > 0)
        .unwrap_or(false)
}

#[cfg(not(test))]
async fn late_package_retry_loop(state: AppState) {
    let mut empty_cycles = 0u64;
    let mut exit = LatePackageRetryExit::Idle;
    loop {
        match retry_late_resource_packages_once(&state, LATE_PACKAGE_MAX_ITEMS).await {
            Ok(report) => {
                if report["state"] == "suspended" {
                    exit = LatePackageRetryExit::Suspended;
                    break;
                }
                if report["lateCount"].as_u64().unwrap_or_default() == 0 {
                    empty_cycles += 1;
                    if empty_cycles >= LATE_PACKAGE_IDLE_STOP_CYCLES {
                        break;
                    }
                } else {
                    empty_cycles = 0;
                }
            }
            Err(err) => {
                eprintln!("late package retry failed: {err}");
            }
        }
        sleep(TokioDuration::from_secs(
            LATE_PACKAGE_RETRY_INTERVAL_SECONDS,
        ))
        .await;
    }
    state
        .late_package_retry_running
        .store(false, Ordering::SeqCst);
    if should_rewake_late_package_retry(exit, has_late_resource_packages(&state).await) {
        wake_late_package_retry_task(state);
    }
}

async fn sync_resources_from_authorized_summary(
    State(state): State<AppState>,
    Json(request): Json<DiscoverySyncRequest>,
) -> ApiResult<Value> {
    if request.items.is_empty() {
        return Err(ApiError::bad_request("empty_authorized_summary_items"));
    }
    let _sync_guard = state.resource_sync_lock.lock().await;
    let cdn_base = state
        .config
        .upstream
        .cdn_endpoint
        .clone()
        .unwrap_or_else(|| state.config.upstream.root_endpoint.clone())
        .trim_end_matches('/')
        .to_owned();
    sync_resources_from_cdn_items(state.clone(), cdn_base, request, Instant::now()).await
}

async fn sync_resources_from_cdn_items(
    state: AppState,
    cdn_base: String,
    request: DiscoverySyncRequest,
    started: Instant,
) -> ApiResult<Value> {
    let start_cursor = read_sync_cursor(&state)
        .await
        .map_err(ApiError::internal)?
        .max(0);
    let target_cursor = request.cursor_hint.unwrap_or_else(|| {
        request
            .items
            .iter()
            .map(|item| item.publication_cursor)
            .max()
            .unwrap_or(start_cursor)
    });
    let max_publications = request.max_publications.unwrap_or(10_000).max(1);
    let mut rejected = Vec::new();
    let mut accepted = Vec::<(i64, ResourcePackage)>::new();
    let mut fetched_count = 0usize;
    let mut cursor = start_cursor;
    let mut blocked_cursor: Option<i64> = None;
    let discovery_domains =
        local_discovery_authorized_domains(&state).map_err(ApiError::internal)?;

    let mut items = request.items;
    items.sort_by_key(|item| item.publication_cursor);
    let candidate_items = items
        .into_iter()
        .take(max_publications)
        .filter(|item| item.publication_cursor > start_cursor)
        .take_while(|item| item.publication_cursor <= target_cursor)
        .collect::<Vec<_>>();
    let batch_packages = fetch_cdn_resource_packages_batch(&state, &cdn_base, &candidate_items)
        .await
        .unwrap_or_default();
    for item in candidate_items {
        if item.publication_cursor <= start_cursor {
            continue;
        }
        if item.publication_cursor > target_cursor {
            break;
        }
        fetched_count += 1;
        cursor = cursor.max(item.publication_cursor);
        let package = if let Some(package) = batch_packages.get(&item.resource_did) {
            package.clone()
        } else {
            match fetch_cdn_resource_package_classified(&state, &cdn_base, &item.resource_did).await
            {
                Ok(PackageFetchOutcome::Found(package)) => *package,
                Ok(PackageFetchOutcome::Unavailable(reason)) => {
                    rejected.push(rejected_package_from_notification(
                        &item,
                        REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE,
                        Some(reason),
                    ));
                    blocked_cursor.get_or_insert(item.publication_cursor);
                    continue;
                }
                Ok(PackageFetchOutcome::PermanentFailure(reason)) => {
                    rejected.push(rejected_package_from_notification(
                        &item,
                        "cdn_fetch_failed",
                        Some(reason),
                    ));
                    blocked_cursor.get_or_insert(item.publication_cursor);
                    continue;
                }
                Err(err) => {
                    rejected.push(rejected_package_from_notification(
                        &item,
                        REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE,
                        Some(err.to_string()),
                    ));
                    blocked_cursor.get_or_insert(item.publication_cursor);
                    continue;
                }
            }
        };
        if package.resource_did != item.resource_did {
            rejected.push(rejected_package_with_reason(&item, "resource_did_mismatch"));
            blocked_cursor.get_or_insert(item.publication_cursor);
            continue;
        }
        if let Err(reason) = validate_notified_resource_package(&item, &package) {
            rejected.push(rejected_package_with_reason(&item, reason));
            blocked_cursor.get_or_insert(item.publication_cursor);
            continue;
        }
        if !authorized_domains_cover(&discovery_domains, &package.metadata.authorized_domains) {
            rejected.push(rejected_package_with_reason(&item, "unauthorized_domains"));
            blocked_cursor.get_or_insert(item.publication_cursor);
            continue;
        }
        if let Err(reason) = validate_resource_package_for_index(&package) {
            rejected.push(rejected_package_with_reason(&item, reason));
            blocked_cursor.get_or_insert(item.publication_cursor);
            continue;
        }
        accepted.push((item.publication_cursor, package));
    }

    let accepted_dids = accepted
        .iter()
        .map(|(_, package)| package.resource_did.as_str())
        .collect::<HashSet<_>>();
    rejected.retain(|item| !accepted_dids.contains(rejected_package_resource_did(item)));
    blocked_cursor = rejected
        .iter()
        .filter_map(|item| item.get("cursor").and_then(Value::as_i64))
        .min();

    let synced = accepted.len();
    upsert_indexed_resource_packages_batch(&state, &accepted)
        .await
        .map_err(ApiError::internal)?;
    write_sync_cursor(&state, cursor)
        .await
        .map_err(ApiError::internal)?;
    let history = json!({
        "syncedAt": Utc::now(),
        "status": "synced",
        "syncMode": "authorized-summary",
        "syncedResourceCount": synced,
        "rejectedCount": rejected.len(),
        "elapsedMs": started.elapsed().as_millis(),
        "fromCursor": start_cursor,
        "toCursor": cursor,
        "targetCursor": target_cursor,
        "pagesFetched": 0,
        "itemsFetched": fetched_count,
        "blockedCursor": blocked_cursor,
        "cursorLag": (target_cursor - cursor).max(0),
        "backend": if state.postgres.is_some() { "postgres" } else if state.sqlite.is_some() { "sqlite" } else { "json" },
        "deltaUpsert": true
    });
    write_sync_history_store(&state, history)
        .await
        .map_err(ApiError::internal)?;
    merge_rejected_packages(&state, &rejected)
        .await
        .map_err(ApiError::internal)?;
    #[cfg(not(test))]
    wake_late_package_retry_task(state.clone());
    Ok(Json(json!({
        "status": "synced",
        "syncMode": "authorized-summary",
        "syncedResourceCount": synced,
        "rejectedCount": rejected.len(),
        "elapsedMs": started.elapsed().as_millis(),
        "fromCursor": start_cursor,
        "toCursor": cursor,
        "targetCursor": target_cursor,
        "pagesFetched": 0,
        "itemsFetched": fetched_count,
        "blockedCursor": blocked_cursor,
        "cursorLag": (target_cursor - cursor).max(0),
        "deltaUpsert": true,
        "rejected": rejected
    })))
}

async fn fetch_cdn_resource_packages_batch(
    state: &AppState,
    cdn_base: &str,
    items: &[DiscoveryNotificationItem],
) -> Result<BTreeMap<String, ResourcePackage>> {
    if items.is_empty() {
        return Ok(BTreeMap::new());
    }
    let resource_dids = items
        .iter()
        .map(|item| item.resource_did.clone())
        .collect::<Vec<_>>();
    let response = state
        .client
        .post(format!("{cdn_base}/cdn/resources/batch-get"))
        .json(&json!({ "resourceDids": resource_dids }))
        .send()
        .await?;
    if !response.status().is_success() {
        return Ok(BTreeMap::new());
    }
    let value: Value = response.json().await?;
    let mut packages = BTreeMap::new();
    for item in value
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let Some(resource_did) = item.get("resourceDid").and_then(Value::as_str) else {
            continue;
        };
        let Some(package_value) = item.get("package") else {
            continue;
        };
        let package = serde_json::from_value::<ResourcePackage>(package_value.clone())?;
        packages.insert(resource_did.to_owned(), package);
    }
    Ok(packages)
}

async fn fetch_cdn_resource_package_classified(
    state: &AppState,
    cdn_base: &str,
    resource_did: &str,
) -> Result<PackageFetchOutcome> {
    let encoded_did =
        url::form_urlencoded::byte_serialize(resource_did.as_bytes()).collect::<String>();
    let response = match state
        .client
        .get(format!("{cdn_base}/cdn/resources/{encoded_did}"))
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) if err.is_timeout() => {
            return Ok(PackageFetchOutcome::Unavailable("cdn_timeout".to_owned()));
        }
        Err(err) if err.is_connect() => {
            return Ok(PackageFetchOutcome::Unavailable(
                "cdn_connect_failed".to_owned(),
            ));
        }
        Err(err) => return Err(err.into()),
    };
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(PackageFetchOutcome::Unavailable("cdn_not_found".to_owned()));
    }
    let status = response.status();
    if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
        return Ok(PackageFetchOutcome::Unavailable(format!(
            "cdn_status_{}",
            status.as_u16()
        )));
    }
    if !status.is_success() {
        return Ok(PackageFetchOutcome::PermanentFailure(format!(
            "cdn_status_{}",
            status.as_u16()
        )));
    }
    match response.json::<ResourcePackage>().await {
        Ok(package) => Ok(PackageFetchOutcome::Found(Box::new(package))),
        Err(err) => Ok(PackageFetchOutcome::PermanentFailure(format!(
            "cdn_decode_failed:{err}"
        ))),
    }
}

fn validate_notified_resource_package(
    item: &DiscoveryNotificationItem,
    package: &ResourcePackage,
) -> std::result::Result<(), String> {
    if package.resource_did != item.resource_did {
        return Err("resource_did_mismatch".to_owned());
    }
    if let Some(resource_type) = item.resource_type.as_deref() {
        let package_resource_type = serde_json::to_value(&package.resource_type)
            .ok()
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_default();
        if package_resource_type != resource_type {
            return Err("resource_type_mismatch".to_owned());
        }
    }
    if !item.capability_tags.is_empty() && package.metadata.capability_tags != item.capability_tags
    {
        return Err("capability_tags_mismatch".to_owned());
    }
    if item.authorized_domains.is_empty() {
        return Err("resource_domains_required".to_owned());
    }
    validate_authorized_domain_list(&item.authorized_domains)?;
    if package.metadata.authorized_domains != item.authorized_domains {
        return Err("authorized_domains_mismatch".to_owned());
    }
    Ok(())
}

fn rejected_package_from_notification(
    item: &DiscoveryNotificationItem,
    reason: impl Into<String>,
    last_error: Option<String>,
) -> Value {
    let reason = reason.into();
    let now = Utc::now().to_rfc3339();
    let mut rejected = json!({
        "resourceDid": item.resource_did,
        "cursor": item.publication_cursor,
        "packageVersion": item.package_version,
        "packageHash": item.package_hash,
        "metadataHash": item.metadata_hash,
        "didDocumentHash": item.did_document_hash,
        "resourceType": item.resource_type,
        "capabilityTags": item.capability_tags,
        "authorizedDomains": item.authorized_domains,
        "reason": reason
    });
    if rejected["reason"] == REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE {
        rejected["status"] = json!(REJECT_STATUS_LATE);
        rejected["retryCount"] = json!(0);
        rejected["firstSeenAt"] = json!(now);
        rejected["lastRetryAt"] = Value::Null;
        rejected["lastError"] = last_error.map(Value::String).unwrap_or(Value::Null);
    }
    rejected
}

fn rejected_package_with_reason(
    item: &DiscoveryNotificationItem,
    reason: impl Into<String>,
) -> Value {
    rejected_package_from_notification(item, reason, None)
}

fn rejected_package_status(item: &Value) -> &str {
    item.get("status")
        .and_then(Value::as_str)
        .unwrap_or(REJECT_STATUS_LATE)
}

fn rejected_package_retry_count(item: &Value) -> u64 {
    item.get("retryCount")
        .and_then(Value::as_u64)
        .unwrap_or_default()
}

fn rejected_package_cursor(item: &Value) -> Option<i64> {
    item.get("cursor").and_then(Value::as_i64)
}

fn rejected_package_version(item: &Value) -> Option<&str> {
    item.get("packageVersion").and_then(Value::as_str)
}

fn is_stale_rejected_for_indexed_did(
    item: &Value,
    resource_did: &str,
    indexed_cursor: i64,
    indexed_version: Option<&str>,
) -> bool {
    if rejected_package_resource_did(item) != resource_did {
        return false;
    }
    if rejected_package_cursor(item).is_some_and(|cursor| cursor <= indexed_cursor) {
        return true;
    }
    if let Some(indexed_version) = indexed_version {
        if !indexed_version.is_empty()
            && rejected_package_version(item).is_some_and(|version| version == indexed_version)
        {
            return true;
        }
    }
    rejected_package_cursor(item).is_none()
}

fn rejected_package_to_notification(item: &Value) -> Option<DiscoveryNotificationItem> {
    Some(DiscoveryNotificationItem {
        resource_did: item
            .get("resourceDid")
            .or_else(|| item.get("did"))
            .and_then(Value::as_str)?
            .to_owned(),
        package_version: item
            .get("packageVersion")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        publication_cursor: item
            .get("cursor")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        package_hash: item
            .get("packageHash")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        metadata_hash: item
            .get("metadataHash")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        did_document_hash: item
            .get("didDocumentHash")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        resource_type: item
            .get("resourceType")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        capability_tags: item
            .get("capabilityTags")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        authorized_domains: item
            .get("authorizedDomains")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
    })
}

fn rejected_package_retry_summary(items: &[Value]) -> Value {
    let late_count = items
        .iter()
        .filter(|item| {
            item.get("reason").and_then(Value::as_str)
                == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
                && rejected_package_status(item) == REJECT_STATUS_LATE
        })
        .count();
    let dead_count = items
        .iter()
        .filter(|item| {
            item.get("reason").and_then(Value::as_str)
                == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
                && rejected_package_status(item) == REJECT_STATUS_DEAD
        })
        .count();
    rejected_package_retry_summary_from_counts(late_count, dead_count)
}

fn rejected_package_retry_summary_from_counts(late_count: usize, dead_count: usize) -> Value {
    json!({
        "enabled": true,
        "state": if late_count > LATE_PACKAGE_MAX_ITEMS {
            "suspended"
        } else if late_count > 0 {
            "running"
        } else {
            "idle"
        },
        "intervalSeconds": LATE_PACKAGE_RETRY_INTERVAL_SECONDS,
        "lateCount": late_count,
        "deadCount": dead_count,
        "consecutiveEmptyCycles": 0,
        "maxLatePackages": LATE_PACKAGE_MAX_ITEMS,
        "suspendedReason": if late_count > LATE_PACKAGE_MAX_ITEMS {
            json!("late_package_limit_exceeded")
        } else {
            Value::Null
        }
    })
}

async fn resource_query(
    State(state): State<AppState>,
    Json(query): Json<ResourceDiscoveryQuery>,
) -> ApiResult<ResourceDiscoveryResponse> {
    let started = Instant::now();
    record_discovery_query_stat(&state)
        .await
        .map_err(ApiError::internal)?;
    if let Some(resource_did) = query_resource_did(&query) {
        let package = read_indexed_resource_package(&state, resource_did)
            .await
            .map_err(ApiError::internal)?;
        let allowed_domains =
            local_discovery_authorized_domains(&state).map_err(ApiError::internal)?;
        let candidates = package
            .filter(|package| {
                resource_matches_exact_did_query(package, &query)
                    && authorized_domains_cover(
                        &allowed_domains,
                        &package.metadata.authorized_domains,
                    )
            })
            .map(|package| discovery_candidate_from_package(package, 1.0))
            .into_iter()
            .collect::<Vec<_>>();
        println!(
            "discovery_query backend=exact-did indexed=true candidates={} returned={} elapsed_ms={}",
            candidates.len(),
            candidates.len(),
            started.elapsed().as_millis()
        );
        return Ok(Json(ResourceDiscoveryResponse {
            discovery_did: state.did,
            candidates,
            created_at: Utc::now(),
            proof: None,
        }));
    }
    let semantic_result = query_semantic_index(&state, &query).await;
    let (mut candidates, candidate_count, prefiltered, backend) = match semantic_result {
        Ok(Some(hits)) if !hits.is_empty() => {
            let candidate_count = hits.len();
            (
                hits.into_iter()
                    .map(|hit| discovery_candidate_from_package(hit.package, hit.final_score))
                    .collect::<Vec<_>>(),
                candidate_count,
                true,
                "semantic-pgvector".to_owned(),
            )
        }
        Ok(Some(_)) => {
            let (candidates, candidate_count, prefiltered) =
                keyword_discovery_candidates(&state, &query)
                    .await
                    .map_err(ApiError::internal)?;
            (
                candidates,
                candidate_count,
                prefiltered,
                discovery_backend_name(&state),
            )
        }
        Ok(None) => {
            let (candidates, candidate_count, prefiltered) =
                keyword_discovery_candidates(&state, &query)
                    .await
                    .map_err(ApiError::internal)?;
            (
                candidates,
                candidate_count,
                prefiltered,
                discovery_backend_name(&state),
            )
        }
        Err(err) => {
            eprintln!("semantic query failed: {err}");
            let (candidates, candidate_count, prefiltered) =
                keyword_discovery_candidates(&state, &query)
                    .await
                    .map_err(ApiError::internal)?;
            (
                candidates,
                candidate_count,
                prefiltered,
                discovery_backend_name(&state),
            )
        }
    };
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(query.limit as usize);
    let metrics = DiscoveryQueryMetrics {
        backend,
        candidate_count,
        returned_count: candidates.len(),
        elapsed_ms: started.elapsed().as_millis(),
        used_indexed_prefilter: prefiltered,
    };
    println!(
        "discovery_query backend={} indexed={} candidates={} returned={} elapsed_ms={}",
        metrics.backend,
        metrics.used_indexed_prefilter,
        metrics.candidate_count,
        metrics.returned_count,
        metrics.elapsed_ms
    );
    Ok(Json(ResourceDiscoveryResponse {
        discovery_did: state.did,
        candidates,
        created_at: Utc::now(),
        proof: None,
    }))
}

async fn api_query_stats(State(state): State<AppState>) -> ApiResult<Value> {
    discovery_query_stats_body(&state)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn discovery_query_stats_body(state: &AppState) -> Result<Value> {
    let generated_at = Utc::now();
    let today = generated_at.date_naive();
    let week_start = today - chrono::Duration::days(today.weekday().num_days_from_monday() as i64);
    let month_start = today.with_day(1).unwrap_or(today);
    let records = read_discovery_query_stats(state).await?;
    let mut today_count = 0i64;
    let mut week_count = 0i64;
    let mut month_count = 0i64;
    for record in records {
        let Ok(stat_date) = chrono::NaiveDate::parse_from_str(&record.stat_date, "%Y-%m-%d") else {
            continue;
        };
        if stat_date == today {
            today_count += record.query_count;
        }
        if stat_date >= week_start {
            week_count += record.query_count;
        }
        if stat_date >= month_start {
            month_count += record.query_count;
        }
    }
    Ok(json!({
        "discoveryDid": state.did,
        "generatedAt": generated_at.to_rfc3339(),
        "windowTimezone": "UTC",
        "todayQueryCount": today_count,
        "weekQueryCount": week_count,
        "monthQueryCount": month_count
    }))
}

async fn run_resource_query(
    state: &AppState,
    query: &ResourceDiscoveryQuery,
) -> Result<ResourceDiscoveryResponse> {
    resource_query(State(state.clone()), Json(query.clone()))
        .await
        .map(|response| response.0)
        .map_err(|err| anyhow!(err.message))
}

async fn evaluate_semantic_discovery(
    state: &AppState,
    suite_path: &Path,
) -> Result<SemanticEvaluationReport> {
    let suite: SemanticEvaluationSuite =
        serde_json::from_str(&read_utf8_text_allowing_bom(suite_path)?)?;
    if suite.cases.is_empty() {
        return Err(anyhow!("semantic_evaluation_suite_empty"));
    }
    let mut reports = Vec::new();
    let mut recall5_hits = 0usize;
    let mut recall10_hits = 0usize;
    let mut reciprocal_rank_sum = 0.0_f32;
    for case in &suite.cases {
        let response = run_resource_query(state, &case.query).await?;
        let returned_dids = response
            .candidates
            .iter()
            .map(|candidate| candidate.resource_did.clone())
            .collect::<Vec<_>>();
        let first_rank = returned_dids
            .iter()
            .position(|did| case.expected_dids.iter().any(|expected| expected == did))
            .map(|index| index + 1);
        let recall_at_5 = first_rank.is_some_and(|rank| rank <= 5);
        let recall_at_10 = first_rank.is_some_and(|rank| rank <= 10);
        if recall_at_5 {
            recall5_hits += 1;
        }
        if recall_at_10 {
            recall10_hits += 1;
        }
        let reciprocal_rank = first_rank
            .filter(|rank| *rank <= 10)
            .map(|rank| 1.0 / rank as f32)
            .unwrap_or(0.0);
        reciprocal_rank_sum += reciprocal_rank;
        reports.push(SemanticEvaluationCaseReport {
            id: case.id.clone(),
            language: case.language.clone(),
            description: case.description.clone(),
            passed: recall_at_10,
            reciprocal_rank,
            returned_dids,
            expected_dids: case.expected_dids.clone(),
        });
    }
    let case_count = reports.len().max(1);
    Ok(SemanticEvaluationReport {
        suite: suite.suite,
        case_count: reports.len(),
        seed_package_count: suite.seed_packages.len() + suite.seed_resources.len(),
        recall_at_5: recall5_hits as f32 / case_count as f32,
        recall_at_10: recall10_hits as f32 / case_count as f32,
        mrr_at_10: reciprocal_rank_sum / case_count as f32,
        cases: reports,
    })
}

fn query_resource_did(query: &ResourceDiscoveryQuery) -> Option<&str> {
    let value = query.query.as_ref()?.trim();
    if value.starts_with("did:oan:") && !value.chars().any(char::is_whitespace) {
        Some(value)
    } else {
        None
    }
}

async fn keyword_discovery_candidates(
    state: &AppState,
    query: &ResourceDiscoveryQuery,
) -> Result<(Vec<ResourceDiscoveryCandidate>, usize, bool)> {
    let (packages, prefiltered) = query_indexed_resource_packages(state, query).await?;
    let allowed_domains = local_discovery_authorized_domains(state)?;
    let packages = packages
        .into_iter()
        .filter(|package| {
            authorized_domains_cover(&allowed_domains, &package.metadata.authorized_domains)
        })
        .collect::<Vec<_>>();
    let candidate_count = packages.len();
    let hits = rank_resource_packages_semantically(packages, query);
    let candidates = hits
        .into_iter()
        .filter(|hit| prefiltered || hit.final_score > 0.0 || query.query.is_none())
        .map(|hit| discovery_candidate_from_package(hit.package, hit.final_score))
        .collect::<Vec<_>>();
    Ok((candidates, candidate_count, prefiltered))
}

fn discovery_candidate_from_package(
    package: ResourcePackage,
    score: f32,
) -> ResourceDiscoveryCandidate {
    ResourceDiscoveryCandidate {
        resource_did: package.resource_did.clone(),
        resource_type: package.resource_type.clone(),
        score,
        version: Some(package.package_version.clone()),
        lifecycle_state: Some(package.metadata.lifecycle_state.clone()),
        capability_tags: package.metadata.capability_tags.clone(),
        authorized_domains: package.metadata.authorized_domains.clone(),
        services: package.metadata.services.clone(),
        protocol_bindings: package.metadata.protocol_bindings.clone(),
        package_info: package
            .did_document
            .oan_metadata
            .as_ref()
            .and_then(|metadata| metadata.package_info.as_ref())
            .and_then(|info| serde_json::to_value(info).ok()),
        root_proof: serde_json::to_value(&package.root_proof).ok(),
    }
}

fn discovery_backend_name(state: &AppState) -> String {
    if state.postgres.is_some() {
        "postgres".to_owned()
    } else if state.sqlite.is_some() {
        "sqlite".to_owned()
    } else {
        "json".to_owned()
    }
}

async fn route_lookup(
    State(state): State<AppState>,
    AxumPath(did): AxumPath<String>,
) -> ApiResult<serde_json::Value> {
    let package = read_indexed_resource_package(&state, &did)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(match package {
        Some(package) => json!({
            "resourceDid": package.resource_did,
            "resourceType": package.resource_type,
            "lifecycleState": package.metadata.lifecycle_state,
            "services": package.metadata.services
        }),
        None => json!({"resourceDid": did, "status": "not-found"}),
    }))
}

async fn api_status(State(state): State<AppState>) -> ApiResult<Value> {
    let indexed_resource_count = count_indexed_resource_packages(&state)
        .await
        .map_err(ApiError::internal)?;
    let last_sync = read_latest_sync_history_item(&state)
        .await
        .map_err(ApiError::internal)?;
    let bulletin = fetch_bulletin(&state).await.ok();
    Ok(Json(json!({
        "discoveryDid": state.did,
        "rootEndpoint": state.config.upstream.root_endpoint,
        "cdnEndpoint": state.config.upstream.cdn_endpoint,
        "indexedResourceCount": indexed_resource_count,
        "lastSync": last_sync,
        "rootAuthorizationStatus": bulletin.as_ref().map(|b| discovery_authorization_status(b, &state.did)).unwrap_or_else(|| "unknown".to_owned()),
        "semanticSearch": semantic_status_json(&state, false, None)
    })))
}

async fn api_root_authorization(State(state): State<AppState>) -> ApiResult<Value> {
    let bulletin = fetch_bulletin(&state).await;
    match bulletin {
        Ok(bulletin) => Ok(Json(json!({
            "discoveryDid": state.did,
            "rootReachable": true,
            "status": discovery_authorization_status(&bulletin, &state.did),
            "authorizedDomains": discovery_authorized_domains(&bulletin, &state.did)
        }))),
        Err(err) => Ok(Json(json!({
            "discoveryDid": state.did,
            "rootReachable": false,
            "status": "unknown",
            "error": err.to_string()
        }))),
    }
}

async fn api_authorized_domains(State(state): State<AppState>) -> ApiResult<Value> {
    let bulletin = fetch_bulletin(&state).await.map_err(ApiError::internal)?;
    let domains = discovery_authorized_domains(&bulletin, &state.did);
    Ok(Json(json!({
        "discoveryDid": state.did,
        "authorizedDomains": domains
    })))
}

async fn api_sync_history(
    State(state): State<AppState>,
    Query(query): Query<DiscoveryPageQuery>,
) -> ApiResult<Value> {
    let limit = validated_discovery_page_limit(query.limit)?;
    let page = read_sync_history_page(&state, query.after_key.as_deref(), limit)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "items": page.items,
        "count": page.items.len(),
        "afterKey": query.after_key,
        "nextKey": page.next_key,
        "hasMore": page.has_more
    })))
}

async fn api_index_stats(State(state): State<AppState>) -> ApiResult<Value> {
    discovery_index_stats_body(&state)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn discovery_index_stats_body(state: &AppState) -> Result<Value> {
    let generated_at = Utc::now();
    if let Ok(cache) = state.index_stats_cache.try_lock() {
        if let Some(cached) = cache.as_ref() {
            if cached.captured_at.elapsed()
                <= StdDuration::from_millis(DISCOVERY_INDEX_STATS_CACHE_TTL_MS)
            {
                return Ok(cached.body.clone());
            }
        }
    }
    let sync_cursor = read_sync_cursor(state).await?;
    let mut body = if let Some(postgres) = &state.postgres {
        let total_row = sqlx::query(&format!("SELECT COUNT(*) FROM {DISCOVERY_PACKAGE_TABLE}"))
            .fetch_one(postgres.pool())
            .await?;
        let type_rows = sqlx::query(&format!(
            "SELECT resource_type, COUNT(*) FROM {DISCOVERY_PACKAGE_TABLE} GROUP BY resource_type"
        ))
        .fetch_all(postgres.pool())
        .await?;
        let tag_rows = sqlx::query(&format!(
            "SELECT tag, COUNT(*) FROM {DISCOVERY_PACKAGE_TABLE}, unnest(capability_tags) AS tag GROUP BY tag"
        ))
        .fetch_all(postgres.pool())
        .await?;
        let mut resource_type_counts = serde_json::Map::new();
        for row in type_rows {
            resource_type_counts.insert(row.get::<String, _>(0), json!(row.get::<i64, _>(1)));
        }
        let mut tag_counts = serde_json::Map::new();
        for row in tag_rows {
            tag_counts.insert(row.get::<String, _>(0), json!(row.get::<i64, _>(1)));
        }
        json!({
            "indexedResourceCount": total_row.get::<i64, _>(0),
            "resourceTypeCounts": resource_type_counts,
            "capabilityTagCounts": tag_counts,
            "backend": "postgres",
            "indexedStats": true,
            "syncCursor": sync_cursor
        })
    } else {
        let packages = read_indexed_resource_packages(state).await?;
        let mut tag_counts = serde_json::Map::new();
        let mut resource_type_counts = serde_json::Map::new();
        for package in &packages {
            let resource_type = package.resource_type.as_str();
            let next_type = resource_type_counts
                .get(resource_type)
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + 1;
            resource_type_counts.insert(resource_type.to_owned(), json!(next_type));
            for tag in &package.metadata.capability_tags {
                let next = tag_counts.get(tag).and_then(Value::as_u64).unwrap_or(0) + 1;
                tag_counts.insert(tag.clone(), json!(next));
            }
        }
        json!({
            "indexedResourceCount": packages.len(),
            "resourceTypeCounts": resource_type_counts,
            "capabilityTagCounts": tag_counts,
            "backend": if state.sqlite.is_some() { "sqlite" } else { "json" },
            "indexedStats": state.sqlite.is_some(),
            "syncCursor": sync_cursor
        })
    };
    if let Value::Object(map) = &mut body {
        map.insert("generatedAt".to_owned(), json!(generated_at.to_rfc3339()));
        map.insert("windowTimezone".to_owned(), json!("UTC"));
        map.insert(
            "semanticSearch".to_owned(),
            semantic_status_json(state, false, None),
        );
        map.insert(
            "semanticIndexedCount".to_owned(),
            match count_semantic_indexed_resources(state).await {
                Ok(Some(count)) => json!(count),
                Ok(None) => Value::Null,
                Err(err) => json!({ "error": err.to_string() }),
            },
        );
    }
    if let Ok(mut cache) = state.index_stats_cache.try_lock() {
        *cache = Some(CachedIndexStats {
            captured_at: Instant::now(),
            body: body.clone(),
        });
    }
    Ok(body)
}

async fn discovery_stats_report_loop(state: AppState) {
    loop {
        if let Err(err) = report_discovery_stats_once(&state).await {
            eprintln!("discovery stats report failed: {err}");
        }
        sleep(TokioDuration::from_secs(
            state.config.stats_report.interval_seconds,
        ))
        .await;
    }
}

async fn report_discovery_stats_once(state: &AppState) -> Result<()> {
    let config = &state.config.stats_report;
    let query_stats = discovery_query_stats_body(state).await?;
    let index_stats = discovery_index_stats_body(state).await?;
    let endpoint = config
        .endpoint
        .as_deref()
        .ok_or_else(|| anyhow!("stats_report.endpoint is required"))?;
    let token = config
        .token
        .as_deref()
        .ok_or_else(|| anyhow!("stats_report.token is required"))?;
    let type_counts = &index_stats["resourceTypeCounts"];
    let indexed_resource_count = json_number_or_null(&index_stats["indexedResourceCount"]);
    let payload = json!({
        "nodeId": config
            .node_id
            .as_deref()
            .ok_or_else(|| anyhow!("stats_report.node_id is required"))?,
        "role": "discovery",
        "status": "online",
        "generatedAt": index_stats["generatedAt"],
        "resourceTotals": {
            "visibleResources": indexed_resource_count.clone(),
            "indexedResources": indexed_resource_count
        },
        "resourcesByType": {
            "agentService": json_number_or_zero(&type_counts["agent_service"]),
            "skill": json_number_or_zero(&type_counts["skill"]),
            "mcpServer": json_number_or_zero(&type_counts["mcp_server"]),
            "toolApi": json_number_or_zero(&type_counts["tool_api"])
        },
        "discoveryCallWindows": {
            "today": json_number_or_zero(&query_stats["todayQueryCount"]),
            "thisWeek": json_number_or_zero(&query_stats["weekQueryCount"]),
            "thisMonth": json_number_or_zero(&query_stats["monthQueryCount"])
        },
        "indexState": {
            "lastIndexedAt": index_stats["generatedAt"],
            "semanticEnabled": index_stats["semanticSearch"]["enabled"],
            "semanticIndexedResources": json_number_or_null(&index_stats["semanticIndexedCount"])
        }
    });
    let response = state
        .client
        .post(endpoint)
        .bearer_auth(token)
        .json(&payload)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("homepage returned {status}: {body}"));
    }
    Ok(())
}

fn json_number_or_zero(value: &Value) -> Value {
    json!(nonnegative_json_number(value).unwrap_or(0))
}

fn json_number_or_null(value: &Value) -> Value {
    nonnegative_json_number(value).map_or(Value::Null, |number| json!(number))
}

fn nonnegative_json_number(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_i64()
            .and_then(|number| (number >= 0).then_some(number as u64))
    })
}

async fn api_index_resources(
    State(state): State<AppState>,
    Query(query): Query<DiscoveryPageQuery>,
) -> ApiResult<Value> {
    let limit = validated_discovery_page_limit(query.limit)?;
    let after_cursor = query.after_cursor.unwrap_or(0).max(0);
    let page = read_indexed_resource_summary_page(
        &state,
        after_cursor,
        query.after_resource_did.as_deref(),
        limit,
    )
    .await
    .map_err(|error| match error.to_string().as_str() {
        "discovery_index_package_too_large" | "discovery_index_page_too_large" => {
            ApiError::bad_request(error.to_string())
        }
        _ => ApiError::internal(error),
    })?;
    Ok(Json(json!({
        "items": page.items,
        "count": page.items.len(),
        "afterCursor": after_cursor,
        "afterResourceDid": query.after_resource_did,
        "nextCursor": page.next_cursor,
        "nextResourceDid": page.next_resource_did,
        "hasMore": page.has_more
    })))
}

async fn api_index_resource_visibility(
    State(state): State<AppState>,
    Json(request): Json<IndexedResourceVisibilityRequest>,
) -> ApiResult<Value> {
    let visible = indexed_resource_visibility(&state, &request.resource_dids)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "resourceDids": request.resource_dids,
        "visible": visible,
        "visibleCount": visible.len()
    })))
}

async fn api_index_resource_detail(
    State(state): State<AppState>,
    AxumPath(did): AxumPath<String>,
) -> ApiResult<Value> {
    let package = read_indexed_resource_package(&state, &did)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "resourceDid": did, "package": package })))
}

async fn api_query_suggestions(
    State(state): State<AppState>,
    Json(input): Json<DiscoverySuggestionInput>,
) -> ApiResult<Value> {
    let result = state
        .recommender
        .suggest_discovery_query(
            input,
            DiscoverySuggestionContext {
                discovery_did: state.did.clone(),
                searchable_domains: local_discovery_authorized_domains(&state)
                    .map_err(ApiError::internal)?,
                max_authorized_domain_hints: 8,
                max_capability_tag_candidates: 12,
            },
        )
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    Ok(Json(
        serde_json::to_value(result).map_err(|err| ApiError::internal(anyhow!(err)))?,
    ))
}

async fn api_query_explain(
    State(state): State<AppState>,
    Json(query): Json<ResourceDiscoveryQuery>,
) -> ApiResult<Value> {
    if state.config.semantic_search.explain_enabled {
        match query_semantic_index(&state, &query).await {
            Ok(Some(hits)) if !hits.is_empty() => {
                let items = hits
                    .iter()
                    .map(|hit| {
                        json!({
                            "resourceDid": hit.package.resource_did,
                            "resourceType": hit.package.resource_type,
                            "matched": true,
                            "score": hit.final_score,
                            "semanticScore": hit.semantic_score,
                            "bm25Score": hit.bm25_score,
                            "tagScore": hit.tag_score,
                            "contextScore": hit.context_score,
                            "intentMaxSimScore": hit.intent_score,
                            "structuredScore": hit.structured_score,
                            "finalScore": hit.final_score,
                            "matchedIntent": hit.matched_intent,
                            "capabilityTagOverlap": query.capability_tags.iter()
                                .filter(|tag| hit.package.metadata.capability_tags.iter().any(|candidate| candidate == *tag))
                                .cloned()
                                .collect::<Vec<_>>(),
                            "resourceTypeMatched": query.resource_type.as_ref().map(|resource_type| {
                                resource_type == &hit.package.resource_type
                            }),
                            "protocolMatched": query.protocol.as_ref().map(|protocol| {
                                hit.package.metadata.services.iter().any(|service| {
                                    service
                                        .protocol
                                        .as_ref()
                                        .map(|candidate| candidate.eq_ignore_ascii_case(protocol))
                                        .unwrap_or(false)
                                }) || hit.package.metadata.protocol_bindings.iter().any(|binding| {
                                    binding["protocol"]
                                        .as_str()
                                        .map(|candidate| candidate.eq_ignore_ascii_case(protocol))
                                        .unwrap_or(false)
                                })
                            }),
                            "matchReason": semantic_match_reason(hit)
                        })
                    })
                    .collect::<Vec<_>>();
                return Ok(Json(json!({
                    "query": query,
                    "items": items,
                    "candidateCount": items.len(),
                    "usedIndexedPrefilter": true,
                    "semanticSearch": semantic_status_json(&state, false, None)
                })));
            }
            Ok(Some(_)) => {}
            Ok(None) => {}
            Err(err) => {
                eprintln!("semantic explain failed: {err}");
            }
        }
    }
    let (packages, prefiltered) = query_indexed_resource_packages(&state, &query)
        .await
        .map_err(ApiError::internal)?;
    let hits = rank_resource_packages_semantically(packages, &query);
    let explanations = hits
        .iter()
        .map(|hit| {
            let package = &hit.package;
            let matched = hit.final_score > 0.0 || query.query.is_none();
            json!({
                "resourceDid": package.resource_did,
                "resourceType": package.resource_type,
                "matched": matched,
                "score": if matched { hit.final_score } else { 0.0 },
                "textMatched": query.query.as_ref().map(|text| {
                    let needle = text.to_ascii_lowercase();
                    package.resource_did.to_ascii_lowercase().contains(&needle)
                        || package.metadata.name.to_ascii_lowercase().contains(&needle)
                        || package.metadata.description.to_ascii_lowercase().contains(&needle)
                        || package.metadata.capability_tags.iter().any(|tag| tag.to_ascii_lowercase().contains(&needle))
                }),
                "capabilityTagOverlap": query.capability_tags.iter()
                    .filter(|tag| package.metadata.capability_tags.iter().any(|candidate| candidate == *tag))
                    .cloned()
                    .collect::<Vec<_>>(),
                "resourceTypeMatched": query.resource_type.as_ref().map(|resource_type| {
                    resource_type == &package.resource_type
                }),
                "protocolMatched": query.protocol.as_ref().map(|protocol| {
                    package.metadata.services.iter().any(|service| {
                        service
                            .protocol
                            .as_ref()
                            .map(|candidate| candidate.eq_ignore_ascii_case(protocol))
                            .unwrap_or(false)
                    })
                }),
                "semanticScore": hit.semantic_score,
                "bm25Score": hit.bm25_score,
                "tagScore": hit.tag_score,
                "contextScore": hit.context_score,
                "intentMaxSimScore": hit.intent_score,
                "structuredScore": hit.structured_score,
                "finalScore": hit.final_score,
                "matchedIntent": hit.matched_intent,
                "matchReason": semantic_match_reason(hit),
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "query": query,
        "items": explanations,
        "candidateCount": explanations.len(),
        "usedIndexedPrefilter": prefiltered,
        "semanticSearch": semantic_status_json(&state, true, Some("keyword_explain_fallback".to_owned()))
    })))
}

async fn api_rejected_packages(
    State(state): State<AppState>,
    Query(query): Query<DiscoveryPageQuery>,
) -> ApiResult<Value> {
    let limit = validated_discovery_page_limit(query.limit)?;
    let page = read_rejected_packages_page(
        &state,
        query.after_key.as_deref(),
        limit,
        query.status.as_deref(),
    )
    .await
    .map_err(ApiError::internal)?;
    let retry = rejected_package_retry_summary_from_store(&state)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(json!({
        "items": page.items,
        "count": page.items.len(),
        "afterKey": query.after_key,
        "nextKey": page.next_key,
        "hasMore": page.has_more,
        "retry": retry
    })))
}

async fn api_backfill_rejected_packages(
    State(state): State<AppState>,
    Json(request): Json<RejectedPackageBackfillRequest>,
) -> ApiResult<Value> {
    let _sync_guard = state.resource_sync_lock.lock().await;
    let cdn_base = state
        .config
        .upstream
        .cdn_endpoint
        .clone()
        .unwrap_or_else(|| state.config.upstream.root_endpoint.clone())
        .trim_end_matches('/')
        .to_owned();
    backfill_rejected_resource_packages(&state, &cdn_base, request)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn api_audit_resolved_rejected_packages(
    State(state): State<AppState>,
    Json(request): Json<ResolvedRejectedPackageCleanupRequest>,
) -> ApiResult<Value> {
    audit_resolved_rejected_packages(&state, request)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn backfill_rejected_resource_packages(
    state: &AppState,
    cdn_base: &str,
    request: RejectedPackageBackfillRequest,
) -> Result<Value> {
    let max_items = request.max_items.unwrap_or(100).max(1);
    let requested_dids = request.resource_dids.into_iter().collect::<BTreeSet<_>>();
    let mut attempted = 0usize;
    let mut recovered = Vec::<Value>::new();
    let mut failed = Vec::<Value>::new();
    let mut skipped = 0usize;
    let mut after_key = None::<String>;
    loop {
        let page = read_rejected_packages_page(
            state,
            after_key.as_deref(),
            DISCOVERY_DEFAULT_PAGE_SIZE,
            None,
        )
        .await?;
        if page.items.is_empty() {
            break;
        }
        for item in page.items {
            let reason = item
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let resource_did = item
                .get("resourceDid")
                .or_else(|| item.get("did"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if reason != REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE
                || rejected_package_status(&item) == REJECT_STATUS_DEAD
                || resource_did.is_empty()
                || (!requested_dids.is_empty() && !requested_dids.contains(resource_did))
            {
                skipped += 1;
                continue;
            }
            if attempted >= max_items {
                skipped += 1;
                continue;
            }
            attempted += 1;
            let cursor = item
                .get("cursor")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            match recover_rejected_package_once(state, cdn_base, &item).await? {
                RejectedPackageRecoveryResult::Recovered => {
                    recovered.push(json!({
                        "resourceDid": resource_did,
                        "cursor": cursor
                    }));
                }
                RejectedPackageRecoveryResult::StillLate(updated)
                | RejectedPackageRecoveryResult::Dead(updated) => failed.push(json!({
                    "resourceDid": resource_did,
                    "cursor": cursor,
                    "reason": REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE,
                    "status": updated["status"],
                    "retryCount": updated["retryCount"],
                    "lastError": updated["lastError"]
                })),
                RejectedPackageRecoveryResult::ReplacementRejected(updated) => failed.push(json!({
                    "resourceDid": resource_did,
                    "cursor": cursor,
                    "reason": updated["reason"],
                    "status": updated["status"],
                    "lastError": updated["lastError"]
                })),
            }
        }
        if !page.has_more {
            break;
        }
        after_key = page.next_key;
    }

    let remaining_rejected_count = count_rejected_packages(state).await?;

    Ok(json!({
        "status": "backfilled",
        "reason": "resource_package_unavailable",
        "attemptedCount": attempted,
        "recoveredCount": recovered.len(),
        "skippedCount": skipped,
        "failedCount": failed.len(),
        "remainingRejectedCount": remaining_rejected_count,
        "recovered": recovered,
        "failed": failed
    }))
}

async fn retry_late_resource_packages_once(state: &AppState, max_items: usize) -> Result<Value> {
    let _sync_guard = state.resource_sync_lock.lock().await;
    let late_items = read_late_rejected_packages(state, LATE_PACKAGE_MAX_ITEMS + 1).await?;
    if late_items.len() > LATE_PACKAGE_MAX_ITEMS {
        return Ok(json!({
            "state": "suspended",
            "lateCount": late_items.len(),
            "attemptedCount": 0,
            "recoveredCount": 0,
            "failedCount": 0,
            "suspendedReason": "late_package_limit_exceeded"
        }));
    }
    let cdn_base = state
        .config
        .upstream
        .cdn_endpoint
        .clone()
        .unwrap_or_else(|| state.config.upstream.root_endpoint.clone())
        .trim_end_matches('/')
        .to_owned();
    let mut attempted = 0usize;
    let mut recovered = 0usize;
    let mut failed = 0usize;
    for item in late_items.into_iter().take(max_items.max(1)) {
        attempted += 1;
        match recover_rejected_package_once(state, &cdn_base, &item).await? {
            RejectedPackageRecoveryResult::Recovered => recovered += 1,
            RejectedPackageRecoveryResult::StillLate(_)
            | RejectedPackageRecoveryResult::Dead(_)
            | RejectedPackageRecoveryResult::ReplacementRejected(_) => failed += 1,
        }
    }
    let late_count = count_late_rejected_packages(state).await?;
    Ok(json!({
        "state": if late_count > 0 { "running" } else { "idle" },
        "lateCount": late_count,
        "attemptedCount": attempted,
        "recoveredCount": recovered,
        "failedCount": failed
    }))
}

async fn read_late_rejected_packages(state: &AppState, limit: usize) -> Result<Vec<Value>> {
    let limit = limit.max(1);
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query(&format!(
            r#"
            SELECT item_json
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE json_extract(item_json, '$.reason') = ?
              AND COALESCE(json_extract(item_json, '$.status'), ?) = ?
            ORDER BY reject_key
            LIMIT ?
            "#
        ))
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .bind(REJECT_STATUS_LATE)
        .bind(REJECT_STATUS_LATE)
        .bind(limit as i64)
        .fetch_all(sqlite.pool())
        .await?;
        return rows
            .into_iter()
            .map(|row| serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(Into::into))
            .collect();
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            r#"
            SELECT item_json::text
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE item_json->>'reason' = $1
              AND COALESCE(item_json->>'status', $2) = $2
            ORDER BY reject_key
            LIMIT $3
            "#
        ))
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .bind(REJECT_STATUS_LATE)
        .bind(limit as i64)
        .fetch_all(postgres.pool())
        .await?;
        return rows
            .into_iter()
            .map(|row| serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(Into::into))
            .collect();
    }
    Ok(read_rejected_packages(state)
        .await?
        .into_iter()
        .filter(|item| {
            item.get("reason").and_then(Value::as_str)
                == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
                && rejected_package_status(item) == REJECT_STATUS_LATE
        })
        .take(limit)
        .collect())
}

async fn count_late_rejected_packages(state: &AppState) -> Result<usize> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!(
            r#"
            SELECT COUNT(*)
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE json_extract(item_json, '$.reason') = ?
              AND COALESCE(json_extract(item_json, '$.status'), ?) = ?
            "#
        ))
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .bind(REJECT_STATUS_LATE)
        .bind(REJECT_STATUS_LATE)
        .fetch_one(sqlite.pool())
        .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!(
            r#"
            SELECT COUNT(*)
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE item_json->>'reason' = $1
              AND COALESCE(item_json->>'status', $2) = $2
            "#
        ))
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .bind(REJECT_STATUS_LATE)
        .fetch_one(postgres.pool())
        .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    Ok(read_rejected_packages(state)
        .await?
        .into_iter()
        .filter(|item| {
            item.get("reason").and_then(Value::as_str)
                == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
                && rejected_package_status(item) == REJECT_STATUS_LATE
        })
        .count())
}

async fn count_rejected_packages(state: &AppState) -> Result<usize> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!("SELECT COUNT(*) FROM {DISCOVERY_REJECTED_TABLE}"))
            .fetch_one(sqlite.pool())
            .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!("SELECT COUNT(*) FROM {DISCOVERY_REJECTED_TABLE}"))
            .fetch_one(postgres.pool())
            .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    Ok(state
        .index
        .read::<Vec<Value>>("rejected-packages.json")
        .unwrap_or_default()
        .len())
}

enum RejectedPackageRecoveryResult {
    Recovered,
    StillLate(Value),
    Dead(Value),
    ReplacementRejected(Value),
}

async fn recover_rejected_package_once(
    state: &AppState,
    cdn_base: &str,
    item: &Value,
) -> Result<RejectedPackageRecoveryResult> {
    let resource_did = rejected_package_resource_did(item).to_owned();
    let cursor = item
        .get("cursor")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    if resource_did.is_empty() || resource_did == "unknown" {
        let replacement = update_rejected_package_validation_failure(
            item,
            "missing_resource_did",
            "missing_resource_did",
        );
        write_rejected_packages(state, std::slice::from_ref(&replacement)).await?;
        return Ok(RejectedPackageRecoveryResult::ReplacementRejected(
            replacement,
        ));
    }
    if let Some(indexed) = read_indexed_resource_package(state, &resource_did).await? {
        let indexed_cursor = read_indexed_resource_cursor(state, &resource_did)
            .await?
            .unwrap_or_default();
        if indexed_cursor >= cursor || indexed.package_version == item["packageVersion"] {
            delete_stale_rejected_packages_for_indexed_did(
                state,
                &resource_did,
                indexed_cursor,
                Some(&indexed.package_version),
            )
            .await?;
            return Ok(RejectedPackageRecoveryResult::Recovered);
        }
    }
    let Some(notification) = rejected_package_to_notification(item) else {
        let replacement = update_rejected_package_validation_failure(
            item,
            "missing_notification_context",
            "missing_notification_context",
        );
        write_rejected_packages(state, std::slice::from_ref(&replacement)).await?;
        return Ok(RejectedPackageRecoveryResult::ReplacementRejected(
            replacement,
        ));
    };
    if notification.package_hash.is_empty()
        || notification.metadata_hash.is_empty()
        || notification.did_document_hash.is_empty()
        || notification.authorized_domains.is_empty()
    {
        let replacement = update_rejected_package_validation_failure(
            item,
            "missing_notification_context",
            "missing_notification_context",
        );
        write_rejected_packages(state, std::slice::from_ref(&replacement)).await?;
        return Ok(RejectedPackageRecoveryResult::ReplacementRejected(
            replacement,
        ));
    }
    match fetch_cdn_resource_package_classified(state, cdn_base, &resource_did).await? {
        PackageFetchOutcome::Found(package) => {
            let package = *package;
            let discovery_domains = local_discovery_authorized_domains(state)?;
            let reject_reason = if package.resource_did != resource_did {
                Some("resource_did_mismatch".to_owned())
            } else if let Err(reason) = validate_notified_resource_package(&notification, &package)
            {
                Some(reason)
            } else if !authorized_domains_cover(
                &discovery_domains,
                &package.metadata.authorized_domains,
            ) {
                Some("unauthorized_domains".to_owned())
            } else {
                validate_resource_package_for_index(&package).err()
            };
            if let Some(reason) = reject_reason {
                let replacement =
                    update_rejected_package_validation_failure(item, &reason, &reason);
                write_rejected_packages(state, std::slice::from_ref(&replacement)).await?;
                return Ok(RejectedPackageRecoveryResult::ReplacementRejected(
                    replacement,
                ));
            }
            let package_version = package.package_version.clone();
            upsert_indexed_resource_package_without_reject_cleanup(state, cursor, package).await?;
            delete_stale_rejected_packages_for_indexed_did(
                state,
                &resource_did,
                cursor,
                Some(&package_version),
            )
            .await?;
            Ok(RejectedPackageRecoveryResult::Recovered)
        }
        PackageFetchOutcome::Unavailable(reason) => {
            let updated = update_rejected_package_retry_failure(item, &reason);
            write_rejected_packages(state, std::slice::from_ref(&updated)).await?;
            if rejected_package_status(&updated) == REJECT_STATUS_DEAD {
                Ok(RejectedPackageRecoveryResult::Dead(updated))
            } else {
                Ok(RejectedPackageRecoveryResult::StillLate(updated))
            }
        }
        PackageFetchOutcome::PermanentFailure(reason) => {
            let replacement =
                update_rejected_package_validation_failure(item, "cdn_fetch_failed", &reason);
            write_rejected_packages(state, std::slice::from_ref(&replacement)).await?;
            Ok(RejectedPackageRecoveryResult::ReplacementRejected(
                replacement,
            ))
        }
    }
}

fn update_rejected_package_retry_failure(item: &Value, last_error: &str) -> Value {
    let mut updated = item.clone();
    let retry_count = rejected_package_retry_count(item) + 1;
    updated["status"] = json!(if retry_count >= LATE_PACKAGE_MAX_RETRY_COUNT {
        REJECT_STATUS_DEAD
    } else {
        REJECT_STATUS_LATE
    });
    updated["retryCount"] = json!(retry_count);
    updated["lastRetryAt"] = json!(Utc::now().to_rfc3339());
    updated["lastError"] = json!(last_error);
    if updated.get("firstSeenAt").and_then(Value::as_str).is_none() {
        updated["firstSeenAt"] = json!(Utc::now().to_rfc3339());
    }
    updated
}

fn update_rejected_package_validation_failure(
    item: &Value,
    reason: &str,
    last_error: &str,
) -> Value {
    let mut updated = item.clone();
    updated["reason"] = json!(reason);
    updated["status"] = Value::Null;
    updated["lastRetryAt"] = json!(Utc::now().to_rfc3339());
    updated["lastError"] = json!(last_error);
    updated
}

async fn audit_resolved_rejected_packages(
    state: &AppState,
    request: ResolvedRejectedPackageCleanupRequest,
) -> Result<Value> {
    let max_items = request.max_items.unwrap_or(1_000).max(1);
    let requested_dids = request.resource_dids.into_iter().collect::<BTreeSet<_>>();
    let mut resolved = Vec::new();
    let mut reject_keys = Vec::new();
    let mut resource_cursor = 0_i64;
    let mut resource_did = None::<String>;
    while resolved.len() < max_items {
        let page = read_visible_indexed_resource_did_page(
            state,
            resource_cursor,
            resource_did.as_deref(),
            DISCOVERY_DEFAULT_PAGE_SIZE,
        )
        .await?;
        if page.items.is_empty() {
            break;
        }
        let visible_dids = page.items.into_iter().collect::<HashSet<_>>();
        let mut reject_cursor = None::<String>;
        while resolved.len() < max_items {
            let rejected_page = read_rejected_packages_page(
                state,
                reject_cursor.as_deref(),
                DISCOVERY_DEFAULT_PAGE_SIZE,
                None,
            )
            .await?;
            if rejected_page.items.is_empty() {
                break;
            }
            for item in rejected_page.items {
                if resolved.len() >= max_items {
                    break;
                }
                let reason = item
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !is_resolvable_historical_reject_reason(reason) {
                    continue;
                }
                let resource_did = rejected_package_resource_did(&item);
                if resource_did.is_empty()
                    || !visible_dids.contains(resource_did)
                    || (!requested_dids.is_empty() && !requested_dids.contains(resource_did))
                {
                    continue;
                }
                resolved.push(json!({
                    "resourceDid": resource_did,
                    "cursor": item.get("cursor").cloned().unwrap_or(Value::Null),
                    "reason": reason,
                    "rejectKey": rejected_package_key(&item)
                }));
                reject_keys.push(rejected_package_key(&item));
            }
            if !rejected_page.has_more {
                break;
            }
            reject_cursor = rejected_page.next_key;
        }
        if !page.has_more {
            break;
        }
        resource_cursor = page.next_cursor;
        resource_did = page.next_resource_did;
    }

    if !request.dry_run {
        delete_rejected_packages(state, &reject_keys).await?;
    }

    Ok(json!({
        "status": if request.dry_run { "audited" } else { "cleaned" },
        "dryRun": request.dry_run,
        "resolvedRejectedCount": resolved.len(),
        "deletedCount": if request.dry_run { 0 } else { reject_keys.len() },
        "reasons": resolvable_historical_reject_reasons(),
        "items": resolved
    }))
}

fn resolvable_historical_reject_reasons() -> &'static [&'static str] {
    &[
        "package_version_mismatch",
        "capability_tags_mismatch",
        "package_hash_mismatch",
    ]
}

fn is_resolvable_historical_reject_reason(reason: &str) -> bool {
    resolvable_historical_reject_reasons().contains(&reason)
}

async fn api_capability_tree(State(state): State<AppState>) -> ApiResult<Value> {
    let response = state
        .client
        .get(format!(
            "{}/root/capability-tree",
            state.config.upstream.root_endpoint.trim_end_matches('/')
        ))
        .send()
        .await;
    match response {
        Ok(response) if response.status().is_success() => {
            let value = response
                .json()
                .await
                .map_err(|err| ApiError::internal(err.into()))?;
            Ok(Json(value))
        }
        Ok(response) => Ok(Json(json!({
            "source": "root",
            "status": "unavailable",
            "statusCode": response.status().as_u16()
        }))),
        Err(err) => Ok(Json(json!({
            "source": "root",
            "status": "unavailable",
            "error": err.to_string()
        }))),
    }
}

async fn fetch_bulletin(state: &AppState) -> Result<Value> {
    state
        .client
        .get(format!(
            "{}/bulletin",
            state.config.upstream.root_endpoint.trim_end_matches('/')
        ))
        .send()
        .await
        .context("fetch_root_bulletin_failed")?
        .json()
        .await
        .context("decode_root_bulletin_failed")
}

fn discovery_authorization_status(bulletin: &Value, discovery_did: &str) -> String {
    if bulletin["events"]
        .as_array()
        .map(|events| {
            events.iter().any(|event| {
                event["subjectDid"] == discovery_did
                    && event["eventType"] == "DISCOVERY_NODE_REVOKED"
            })
        })
        .unwrap_or(false)
    {
        "revoked".to_owned()
    } else {
        "active".to_owned()
    }
}


async fn upsert_indexed_resource_packages_batch(
    state: &AppState,
    packages: &[(i64, ResourcePackage)],
) -> Result<()> {
    upsert_indexed_resource_packages_batch_with_cleanup(state, packages, true).await
}

async fn upsert_indexed_resource_packages_batch_with_cleanup(
    state: &AppState,
    packages: &[(i64, ResourcePackage)],
    cleanup_rejected_by_did: bool,
) -> Result<()> {
    if packages.is_empty() {
        return Ok(());
    }
    let packages = latest_resource_packages_by_did(packages);
    let accepted_dids = packages
        .iter()
        .map(|(_, package)| package.resource_did.clone())
        .collect::<Vec<_>>();
    let updated_at = Utc::now();
    let updated_at_text = updated_at.to_rfc3339();
    if let Some(sqlite) = &state.sqlite {
        let rows = packages
            .iter()
            .map(|(cursor, package)| {
                Ok::<_, anyhow::Error>((
                    package.resource_did.clone(),
                    *cursor,
                    package.package_version.clone(),
                    serde_json::to_string(package)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tx = sqlite.pool().begin().await?;
        for chunk in rows.chunks(250) {
            let mut builder = QueryBuilder::<Sqlite>::new(format!(
                "INSERT INTO {DISCOVERY_PACKAGE_TABLE}(resource_did, cursor, version, package_json, updated_at) "
            ));
            builder.push_values(
                chunk,
                |mut row, (resource_did, cursor, version, package_json)| {
                    row.push_bind(resource_did)
                        .push_bind(cursor)
                        .push_bind(version)
                        .push_bind(package_json)
                        .push_bind(&updated_at_text);
                },
            );
            builder.push(
                r#"
                ON CONFLICT(resource_did)
                DO UPDATE SET
                    cursor = excluded.cursor,
                    version = excluded.version,
                    package_json = excluded.package_json,
                    updated_at = excluded.updated_at
                "#,
            );
            builder.build().execute(&mut *tx).await?;
        }
        tx.commit().await?;
        if cleanup_rejected_by_did {
            delete_rejected_packages_by_did(state, &accepted_dids).await?;
        }
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        let projected = packages
            .iter()
            .map(|(cursor, package)| {
                Ok::<_, anyhow::Error>(DiscoveryProjectedPackage {
                    cursor: *cursor,
                    package,
                    package_json: serde_json::to_value(package)?,
                    projection: discovery_package_projection(package),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tx = postgres.pool().begin().await?;
        for chunk in projected.chunks(250) {
            let mut builder = QueryBuilder::<Postgres>::new(format!(
                r#"
                INSERT INTO {DISCOVERY_PACKAGE_TABLE}(
                    resource_did, cursor, version, resource_type, lifecycle_state,
                    capability_tags, protocols, service_endpoints, tag_text, search_text,
                    package_json, updated_at
                )
                "#
            ));
            builder.push_values(chunk, |mut row, item| {
                row.push_bind(&item.package.resource_did)
                    .push_bind(item.cursor)
                    .push_bind(&item.package.package_version)
                    .push_bind(&item.projection.resource_type)
                    .push_bind(&item.projection.lifecycle_state)
                    .push_bind(&item.projection.capability_tags)
                    .push_bind(&item.projection.protocols)
                    .push_bind(&item.projection.service_endpoints)
                    .push_bind(&item.projection.tag_text)
                    .push_bind(&item.projection.search_text)
                    .push_bind(&item.package_json)
                    .push_bind(updated_at);
            });
            builder.push(
                r#"
                ON CONFLICT(resource_did)
                DO UPDATE SET
                    cursor = excluded.cursor,
                    version = excluded.version,
                    resource_type = excluded.resource_type,
                    lifecycle_state = excluded.lifecycle_state,
                    capability_tags = excluded.capability_tags,
                    protocols = excluded.protocols,
                    service_endpoints = excluded.service_endpoints,
                    tag_text = excluded.tag_text,
                    search_text = excluded.search_text,
                    package_json = excluded.package_json,
                    updated_at = excluded.updated_at
                "#,
            );
            builder.build().execute(&mut *tx).await?;
        }
        tx.commit().await?;
        if cleanup_rejected_by_did {
            delete_rejected_packages_by_did(state, &accepted_dids).await?;
        }
        if semantic_search_available(state) {
            if let Err(err) = upsert_semantic_index_batch_with_retries(
                state,
                &projected,
                SemanticRebuildStage::Both,
            )
            .await
            {
                eprintln!("semantic index upsert failed: {err}");
            }
        }
        return Ok(());
    }
    let mut indexed = read_indexed_resource_packages(state).await?;
    for (_, package) in packages {
        indexed.retain(|candidate| candidate.resource_did != package.resource_did);
        indexed.push(package.clone());
    }
    state.index.write("resource-capabilities.json", &indexed)?;
    if cleanup_rejected_by_did {
        delete_rejected_packages_by_did(state, &accepted_dids).await?;
    }
    Ok(())
}

async fn upsert_indexed_resource_package_without_reject_cleanup(
    state: &AppState,
    cursor: i64,
    package: ResourcePackage,
) -> Result<()> {
    upsert_indexed_resource_packages_batch_with_cleanup(state, &[(cursor, package)], false).await
}

fn latest_resource_packages_by_did(
    packages: &[(i64, ResourcePackage)],
) -> Vec<(i64, ResourcePackage)> {
    let mut latest = BTreeMap::<String, (i64, ResourcePackage)>::new();
    for (cursor, package) in packages {
        latest
            .entry(package.resource_did.clone())
            .and_modify(|existing| {
                if *cursor >= existing.0 {
                    *existing = (*cursor, package.clone());
                }
            })
            .or_insert_with(|| (*cursor, package.clone()));
    }
    latest.into_values().collect()
}

async fn upsert_semantic_index_batch(
    state: &AppState,
    projected: &[DiscoveryProjectedPackage<'_>],
    stage: SemanticRebuildStage,
) -> Result<()> {
    let engine = DiscoverySearchEngine::from_config(&state.config.semantic_search)?;
    engine.upsert_batch(state, projected, stage).await
}

async fn upsert_semantic_index_batch_with_retries(
    state: &AppState,
    projected: &[DiscoveryProjectedPackage<'_>],
    stage: SemanticRebuildStage,
) -> Result<()> {
    if projected.is_empty() {
        return Ok(());
    }
    match upsert_semantic_index_batch(state, projected, stage).await {
        Ok(()) => {
            for item in projected {
                clear_semantic_rebuild_skip_record(state, stage, &item.package.resource_did)
                    .await?;
            }
            Ok(())
        }
        Err(err) => {
            eprintln!("semantic index batch upsert failed: {err}; retrying individually");
            let mut failed = 0usize;
            for item in projected {
                match upsert_semantic_index_batch(state, std::slice::from_ref(item), stage).await {
                    Ok(()) => {
                        clear_semantic_rebuild_skip_record(
                            state,
                            stage,
                            &item.package.resource_did,
                        )
                        .await?;
                    }
                    Err(item_err) => {
                        failed += 1;
                        store_semantic_rebuild_skip(
                            state,
                            stage,
                            item.cursor,
                            item.package.resource_did.clone(),
                            item_err.to_string(),
                        )
                        .await?;
                    }
                }
            }
            if failed > 0 {
                eprintln!("semantic index single-item retries recorded {failed} skipped packages");
            }
            Ok(())
        }
    }
}

async fn semantic_index_compensation_loop(state: AppState) {
    sleep(TokioDuration::from_secs(60)).await;
    loop {
        if let Err(err) = compensate_semantic_indexes(&state).await {
            eprintln!("semantic index compensation failed: {err}");
        }
        sleep(TokioDuration::from_secs(300)).await;
    }
}

async fn compensate_semantic_indexes(state: &AppState) -> Result<()> {
    if state.postgres.is_none() || !semantic_search_available(state) {
        return Ok(());
    }
    let _sync_guard = state.resource_sync_lock.lock().await;
    backfill_skipped_semantic_indexes(state, SemanticRebuildStage::Both).await?;
    let pending = read_semantic_index_pending_packages(state, 100).await?;
    if pending.is_empty() {
        return Ok(());
    }
    let projected = pending
        .iter()
        .map(|(cursor, package)| DiscoveryProjectedPackage {
            cursor: *cursor,
            package,
            package_json: serde_json::to_value(package).unwrap_or_else(|_| json!({})),
            projection: discovery_package_projection(package),
        })
        .collect::<Vec<_>>();
    upsert_semantic_index_batch_with_retries(state, &projected, SemanticRebuildStage::Both).await
}

async fn upsert_pgvector_semantic_index_batch(
    state: &AppState,
    projected: &[DiscoveryProjectedPackage<'_>],
    stage: SemanticRebuildStage,
) -> Result<()> {
    let Some(postgres) = &state.postgres else {
        return Ok(());
    };
    if !semantic_search_available(state) || projected.is_empty() {
        return Ok(());
    }

    let updated_at = Utc::now();
    let docs = projected
        .iter()
        .map(|item| search_document_from_package(item.cursor, item.package, &item.projection))
        .collect::<Vec<_>>();
    let existing_hashes = read_semantic_source_hashes(state, &docs).await?;

    for chunk in docs.chunks(100) {
        let changed = chunk
            .iter()
            .filter(|doc| {
                existing_hashes
                    .get(&doc.resource_did)
                    .map(|hash| hash != &doc.semantic_source_hash)
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();
        if changed.is_empty() {
            continue;
        }
        let mut embedded = Vec::new();
        for doc in changed {
            let embedding = embed_semantic_text(
                &state.config.semantic_search,
                &state.client,
                &semantic_document_text(doc),
            )
            .await?;
            embedded.push((doc, embedding));
        }

        let mut builder = QueryBuilder::<Postgres>::new(format!(
            r#"
            INSERT INTO {DISCOVERY_SEMANTIC_INDEX_TABLE}(
                resource_did, cursor, package_version, did_document_hash, metadata_hash,
                package_hash, resource_type, lifecycle_state, capability_tags, protocols,
                service_endpoints, name, description, examples, use_cases, search_text,
                tag_text,
                semantic_source_hash, embedding, embedding_model, embedding_version, updated_at
            )
            "#
        ));
        builder.push_values(embedded, |mut row, (doc, embedding)| {
            row.push_bind(&doc.resource_did)
                .push_bind(doc.cursor)
                .push_bind(&doc.package_version)
                .push_bind(&doc.did_document_hash)
                .push_bind(&doc.metadata_hash)
                .push_bind(&doc.package_hash)
                .push_bind(&doc.resource_type)
                .push_bind(&doc.lifecycle_state)
                .push_bind(&doc.capability_tags)
                .push_bind(&doc.protocols)
                .push_bind(&doc.service_endpoints)
                .push_bind(&doc.name)
                .push_bind(&doc.description)
                .push_bind(serde_json::to_value(&doc.examples).unwrap_or_else(|_| json!([])))
                .push_bind(serde_json::to_value(&doc.use_cases).unwrap_or_else(|_| json!([])))
                .push_bind(&doc.tag_text)
                .push_bind(&doc.search_text)
                .push_bind(&doc.semantic_source_hash)
                .push(format!("{}::vector", pgvector_literal(&embedding)))
                .push_bind(&state.semantic.embedding_model)
                .push_bind(&state.semantic.embedding_version)
                .push_bind(updated_at);
        });
        builder.push(
            r#"
            ON CONFLICT(resource_did)
            DO UPDATE SET
                cursor = excluded.cursor,
                package_version = excluded.package_version,
                did_document_hash = excluded.did_document_hash,
                metadata_hash = excluded.metadata_hash,
                package_hash = excluded.package_hash,
                resource_type = excluded.resource_type,
                lifecycle_state = excluded.lifecycle_state,
                capability_tags = excluded.capability_tags,
                protocols = excluded.protocols,
                service_endpoints = excluded.service_endpoints,
                name = excluded.name,
                description = excluded.description,
                examples = excluded.examples,
                use_cases = excluded.use_cases,
                tag_text = excluded.tag_text,
                search_text = excluded.search_text,
                semantic_source_hash = excluded.semantic_source_hash,
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model,
                embedding_version = excluded.embedding_version,
                updated_at = excluded.updated_at
            "#,
        );
        builder.build().execute(postgres.pool()).await?;
    }
    if stage.includes_intent() {
        upsert_pgvector_intent_index_batch(state, &docs).await?;
    }
    Ok(())
}

async fn upsert_pgvector_intent_index_batch(
    state: &AppState,
    docs: &[SearchDocument],
) -> Result<()> {
    let Some(postgres) = &state.postgres else {
        return Ok(());
    };
    if docs.is_empty() {
        return Ok(());
    }
    let updated_at = Utc::now();
    for chunk in docs.chunks(50) {
        let resource_dids = chunk
            .iter()
            .map(|doc| doc.resource_did.clone())
            .collect::<Vec<_>>();
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_INTENT_INDEX_TABLE} WHERE resource_did = ANY($1) AND embedding_model = $2 AND embedding_version = $3"
        ))
        .bind(&resource_dids)
        .bind(&state.semantic.embedding_model)
        .bind(&state.semantic.embedding_version)
        .execute(postgres.pool())
        .await?;

        let mut embedded = Vec::new();
        for doc in chunk {
            for intent in semantic_intent_documents(doc) {
                let embedding = embed_semantic_text(
                    &state.config.semantic_search,
                    &state.client,
                    &intent.intent_text,
                )
                .await?;
                embedded.push((intent, embedding));
            }
        }
        if embedded.is_empty() {
            continue;
        }
        let mut builder = QueryBuilder::<Postgres>::new(format!(
            r#"
            INSERT INTO {DISCOVERY_INTENT_INDEX_TABLE}(
                resource_did, intent_id, intent_kind, intent_text, resource_type,
                lifecycle_state, semantic_source_hash, embedding, embedding_model,
                embedding_version, updated_at
            )
            "#
        ));
        builder.push_values(embedded, |mut row, (intent, embedding)| {
            row.push_bind(intent.resource_did)
                .push_bind(intent.intent_id)
                .push_bind(intent.intent_kind)
                .push_bind(intent.intent_text)
                .push_bind(intent.resource_type)
                .push_bind(intent.lifecycle_state)
                .push_bind(intent.semantic_source_hash)
                .push(format!("{}::vector", pgvector_literal(&embedding)))
                .push_bind(&state.semantic.embedding_model)
                .push_bind(&state.semantic.embedding_version)
                .push_bind(updated_at);
        });
        builder.push(
            r#"
            ON CONFLICT(resource_did, intent_id, embedding_model, embedding_version)
            DO UPDATE SET
                intent_kind = excluded.intent_kind,
                intent_text = excluded.intent_text,
                resource_type = excluded.resource_type,
                lifecycle_state = excluded.lifecycle_state,
                semantic_source_hash = excluded.semantic_source_hash,
                embedding = excluded.embedding,
                updated_at = excluded.updated_at
            "#,
        );
        builder.build().execute(postgres.pool()).await?;
    }
    Ok(())
}

async fn read_semantic_source_hashes(
    state: &AppState,
    docs: &[SearchDocument],
) -> Result<BTreeMap<String, String>> {
    let Some(postgres) = &state.postgres else {
        return Ok(BTreeMap::new());
    };
    if docs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let dids = docs
        .iter()
        .map(|doc| doc.resource_did.clone())
        .collect::<Vec<_>>();
    let rows = sqlx::query(&format!(
        "SELECT resource_did, semantic_source_hash FROM {DISCOVERY_SEMANTIC_INDEX_TABLE} WHERE resource_did = ANY($1) AND embedding_model = $2 AND embedding_version = $3"
    ))
    .bind(&dids)
    .bind(&state.semantic.embedding_model)
    .bind(&state.semantic.embedding_version)
    .fetch_all(postgres.pool())
    .await?;
    Ok(rows
        .into_iter()
        .map(|row| (row.get::<String, _>(0), row.get::<String, _>(1)))
        .collect())
}

async fn count_semantic_indexed_resources(state: &AppState) -> Result<Option<i64>> {
    let Some(postgres) = &state.postgres else {
        return Ok(None);
    };
    if !state.semantic.enabled {
        return Ok(None);
    }
    let row = sqlx::query(&format!(
        "SELECT COUNT(*) FROM {DISCOVERY_SEMANTIC_INDEX_TABLE} WHERE embedding_model = $1 AND embedding_version = $2"
    ))
    .bind(&state.semantic.embedding_model)
    .bind(&state.semantic.embedding_version)
    .fetch_one(postgres.pool())
    .await?;
    Ok(Some(row.get::<i64, _>(0)))
}

async fn count_intent_indexed_rows(state: &AppState) -> Result<Option<i64>> {
    let Some(postgres) = &state.postgres else {
        return Ok(None);
    };
    if !state.semantic.enabled {
        return Ok(None);
    }
    let row = sqlx::query(&format!(
        "SELECT COUNT(*) FROM {DISCOVERY_INTENT_INDEX_TABLE} WHERE embedding_model = $1 AND embedding_version = $2"
    ))
    .bind(&state.semantic.embedding_model)
    .bind(&state.semantic.embedding_version)
    .fetch_one(postgres.pool())
    .await?;
    Ok(Some(row.get::<i64, _>(0)))
}

async fn read_semantic_index_pending_packages(
    state: &AppState,
    limit: i64,
) -> Result<Vec<(i64, ResourcePackage)>> {
    let Some(postgres) = &state.postgres else {
        return Ok(vec![]);
    };
    if !state.semantic.enabled {
        return Ok(vec![]);
    }
    let limit = limit.max(1) as usize;
    let mut pending = Vec::new();
    let mut after_cursor = 0_i64;
    loop {
        let rows = sqlx::query(&format!(
            r#"
            SELECT p.cursor, p.package_json::text, s.semantic_source_hash
            FROM {DISCOVERY_PACKAGE_TABLE} p
            LEFT JOIN {DISCOVERY_SEMANTIC_INDEX_TABLE} s
              ON p.resource_did = s.resource_did
             AND s.embedding_model = $1
             AND s.embedding_version = $2
            WHERE p.cursor > $3
              AND p.lifecycle_state = 'active'
              AND p.resource_type = ANY($4)
            ORDER BY p.cursor, p.resource_did
            LIMIT $5
            "#
        ))
        .bind(&state.semantic.embedding_model)
        .bind(&state.semantic.embedding_version)
        .bind(after_cursor)
        .bind(default_discoverable_resource_type_names())
        .bind(i64::from(DISCOVERY_SEMANTIC_REBUILD_PAGE_SIZE))
        .fetch_all(postgres.pool())
        .await?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            let cursor = row.get::<i64, _>(0);
            after_cursor = cursor;
            let package = serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(1))?;
            let projection = discovery_package_projection(&package);
            let doc = search_document_from_package(cursor, &package, &projection);
            let indexed_hash = row.get::<Option<String>, _>(2);
            if indexed_hash.as_deref() != Some(doc.semantic_source_hash.as_str()) {
                pending.push((cursor, package));
                if pending.len() >= limit {
                    return Ok(pending);
                }
            }
        }
    }
    Ok(pending)
}

async fn clear_semantic_indexes_for_current_model(
    state: &AppState,
    stage: SemanticRebuildStage,
) -> Result<()> {
    let Some(postgres) = &state.postgres else {
        return Ok(());
    };
    if stage.includes_intent() {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_INTENT_INDEX_TABLE} WHERE embedding_model = $1 AND embedding_version = $2"
        ))
        .bind(&state.semantic.embedding_model)
        .bind(&state.semantic.embedding_version)
        .execute(postgres.pool())
        .await?;
    }
    if stage.includes_context() {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_SEMANTIC_INDEX_TABLE} WHERE embedding_model = $1 AND embedding_version = $2"
        ))
        .bind(&state.semantic.embedding_model)
        .bind(&state.semantic.embedding_version)
        .execute(postgres.pool())
        .await?;
    }
    Ok(())
}

async fn count_visible_indexed_resource_packages(state: &AppState) -> Result<usize> {
    let mut count = 0usize;
    let mut cursor = 0_i64;
    let mut resource_did = None::<String>;
    loop {
        let page = read_visible_indexed_resource_page(
            state,
            cursor,
            resource_did.as_deref(),
            DISCOVERY_SEMANTIC_REBUILD_PAGE_SIZE,
        )
        .await?;
        count += page.items.len();
        if !page.has_more {
            break;
        }
        cursor = page.next_cursor;
        resource_did = page.next_resource_did;
    }
    Ok(count)
}

async fn read_visible_indexed_resource_page(
    state: &AppState,
    after_cursor: i64,
    after_resource_did: Option<&str>,
    limit: u32,
) -> Result<CursorPage<(i64, ResourcePackage)>> {
    let discovery_domains = local_discovery_authorized_domains(state)?;
    let mut visible = Vec::new();
    let mut cursor = after_cursor;
    let mut resource_did = after_resource_did.map(ToOwned::to_owned);
    loop {
        let page = read_indexed_resource_page(state, cursor, resource_did.as_deref(), limit.max(1))
            .await?;
        if page.items.is_empty() {
            return Ok(CursorPage {
                items: visible,
                next_cursor: cursor,
                next_resource_did: resource_did,
                has_more: false,
            });
        }
        cursor = page.next_cursor;
        resource_did = page.next_resource_did.clone();
        visible.extend(page.items.into_iter().filter(|(_, package)| {
            package.metadata.lifecycle_state == "active"
                && is_user_consumable_resource_type(&package.resource_type)
                && authorized_domains_cover(
                    &discovery_domains,
                    &package.metadata.authorized_domains,
                )
        }));
        if visible.len() >= limit.max(1) as usize {
            visible.truncate(limit.max(1) as usize);
            let next_cursor = visible.last().map(|(cursor, _)| *cursor).unwrap_or(cursor);
            let next_resource_did = visible
                .last()
                .map(|(_, package)| package.resource_did.clone());
            return Ok(CursorPage {
                items: visible,
                next_cursor,
                next_resource_did,
                has_more: true,
            });
        }
        if !page.has_more {
            return Ok(CursorPage {
                items: visible,
                next_cursor: cursor,
                next_resource_did: resource_did,
                has_more: false,
            });
        }
    }
}

fn projected_discovery_packages<'a>(
    packages: &'a [(i64, ResourcePackage)],
) -> Vec<DiscoveryProjectedPackage<'a>> {
    packages
        .iter()
        .map(|(cursor, package)| DiscoveryProjectedPackage {
            cursor: *cursor,
            package,
            package_json: serde_json::to_value(package).unwrap_or_else(|_| json!({})),
            projection: discovery_package_projection(package),
        })
        .collect()
}

async fn process_semantic_rebuild_batch(
    state: &AppState,
    stage: SemanticRebuildStage,
    page: &[(i64, ResourcePackage)],
    batch_size: usize,
) -> Result<usize> {
    let mut processed = 0usize;
    for chunk in page.chunks(batch_size.max(1)) {
        let projected = projected_discovery_packages(chunk);
        upsert_semantic_index_batch(state, &projected, stage).await?;
        processed += chunk.len();
    }
    Ok(processed)
}

async fn process_semantic_rebuild_skip_batch(
    state: &AppState,
    stage: SemanticRebuildStage,
    page: &[(i64, ResourcePackage)],
    skipped_packages: &mut usize,
) -> Result<usize> {
    let mut processed = 0usize;
    for (cursor, package) in page {
        let single = [(*cursor, package.clone())];
        let projected = projected_discovery_packages(&single);
        match upsert_semantic_index_batch(state, &projected, stage).await {
            Ok(()) => {
                processed += 1;
            }
            Err(err) => {
                *skipped_packages += 1;
                processed += 1;
                store_semantic_rebuild_skip(
                    state,
                    stage,
                    *cursor,
                    package.resource_did.clone(),
                    err.to_string(),
                )
                .await?;
            }
        }
    }
    Ok(processed)
}

async fn rebuild_semantic_indexes(
    state: &AppState,
    reset: bool,
    stage: SemanticRebuildStage,
) -> Result<SemanticRebuildReport> {
    if !semantic_search_available(state) {
        return Ok(SemanticRebuildReport {
            stage: stage.as_str().to_owned(),
            phase: "skipped".to_owned(),
            backend: discovery_backend_name(state),
            semantic_enabled: false,
            package_count: 0,
            context_indexed_count: 0,
            intent_indexed_count: 0,
            skipped_packages: 0,
            recovered_packages: 0,
            retries: 0,
            last_cursor: None,
            safe_to_delete_old_indexes: false,
            skipped_reason: state.semantic.fallback_reason.clone(),
        });
    }
    let progress = if reset {
        None
    } else {
        load_semantic_rebuild_progress(state, stage).await?
    };
    let mut skipped_packages = 0usize;
    let mut retries = 0usize;
    let mut processed = progress
        .as_ref()
        .map(|record| record.processed)
        .unwrap_or(0);
    let total = count_visible_indexed_resource_packages(state).await?;
    let mut cursor = progress
        .as_ref()
        .and_then(|record| record.last_cursor)
        .unwrap_or(0);
    let mut last_cursor = progress.as_ref().and_then(|record| record.last_cursor);
    let mut last_resource_did = progress
        .as_ref()
        .and_then(|record| record.last_resource_did.clone());
    let mut batch_size = state.config.semantic_search.embedding.batch_size.max(1);
    if cursor > 0 {
        skipped_packages = 0;
        retries = 0;
    } else {
        clear_semantic_indexes_for_current_model(state, stage).await?;
        clear_semantic_rebuild_progress(state, stage).await?;
        clear_semantic_rebuild_skip_records(state, Some(stage)).await?;
    }
    loop {
        let page_limit = batch_size
            .min(DISCOVERY_SEMANTIC_REBUILD_PAGE_SIZE as usize)
            .max(1) as u32;
        let page = read_visible_indexed_resource_page(
            state,
            cursor,
            last_resource_did.as_deref(),
            page_limit,
        )
        .await?;
        if page.items.is_empty() {
            break;
        }
        match process_semantic_rebuild_batch(state, stage, &page.items, batch_size).await {
            Ok(done) => {
                processed += done;
                last_cursor = page.items.last().map(|(cursor, _)| *cursor);
                last_resource_did = page.next_resource_did.clone();
                cursor = page.next_cursor;
                store_semantic_rebuild_progress(
                    state,
                    stage,
                    "running",
                    last_cursor,
                    last_resource_did.clone(),
                    processed,
                    total,
                    skipped_packages,
                    retries,
                    batch_size,
                )
                .await?;
            }
            Err(err) => {
                if is_embedding_timeout_error(&err) && batch_size > 1 {
                    retries += 1;
                    batch_size = (batch_size / 2).max(1);
                    store_semantic_rebuild_progress(
                        state,
                        stage,
                        "throttled",
                        last_cursor,
                        last_resource_did.clone(),
                        processed,
                        total,
                        skipped_packages,
                        retries,
                        batch_size,
                    )
                    .await?;
                    continue;
                }
                processed += process_semantic_rebuild_skip_batch(
                    state,
                    stage,
                    &page.items,
                    &mut skipped_packages,
                )
                .await?;
                last_cursor = page.items.last().map(|(cursor, _)| *cursor);
                last_resource_did = page.next_resource_did.clone();
                cursor = page.next_cursor;
                if last_cursor.is_some() {
                    store_semantic_rebuild_progress(
                        state,
                        stage,
                        "skip",
                        last_cursor,
                        last_resource_did.clone(),
                        processed,
                        total,
                        skipped_packages,
                        retries,
                        batch_size,
                    )
                    .await?;
                }
            }
        }
        if !page.has_more {
            break;
        }
    }
    let final_phase = if skipped_packages == 0 {
        "completed"
    } else {
        "completed-with-skips"
    };
    store_semantic_rebuild_progress(
        state,
        stage,
        final_phase,
        last_cursor,
        last_resource_did.clone(),
        processed,
        total,
        skipped_packages,
        retries,
        batch_size,
    )
    .await?;
    Ok(SemanticRebuildReport {
        phase: final_phase.to_owned(),
        stage: stage.as_str().to_owned(),
        backend: discovery_backend_name(state),
        semantic_enabled: true,
        package_count: total,
        context_indexed_count: count_semantic_indexed_resources(state)
            .await?
            .unwrap_or_default() as usize,
        intent_indexed_count: count_intent_indexed_rows(state).await?.unwrap_or_default() as usize,
        skipped_packages,
        recovered_packages: 0,
        retries,
        last_cursor,
        safe_to_delete_old_indexes: skipped_packages == 0
            && matches!(stage, SemanticRebuildStage::Both),
        skipped_reason: None,
    })
}

async fn backfill_skipped_semantic_indexes(
    state: &AppState,
    stage: SemanticRebuildStage,
) -> Result<SemanticRebuildReport> {
    if !semantic_search_available(state) {
        return Ok(SemanticRebuildReport {
            stage: stage.as_str().to_owned(),
            phase: "skipped".to_owned(),
            backend: discovery_backend_name(state),
            semantic_enabled: false,
            package_count: 0,
            context_indexed_count: 0,
            intent_indexed_count: 0,
            skipped_packages: 0,
            recovered_packages: 0,
            retries: 0,
            last_cursor: None,
            safe_to_delete_old_indexes: false,
            skipped_reason: state.semantic.fallback_reason.clone(),
        });
    }

    let skipped = load_semantic_rebuild_skip_records(state, Some(stage)).await?;
    if skipped.is_empty() {
        return Ok(SemanticRebuildReport {
            stage: stage.as_str().to_owned(),
            phase: "completed".to_owned(),
            backend: discovery_backend_name(state),
            semantic_enabled: true,
            package_count: 0,
            context_indexed_count: count_semantic_indexed_resources(state)
                .await?
                .unwrap_or_default() as usize,
            intent_indexed_count: count_intent_indexed_rows(state).await?.unwrap_or_default()
                as usize,
            skipped_packages: 0,
            recovered_packages: 0,
            retries: 0,
            last_cursor: None,
            safe_to_delete_old_indexes: matches!(stage, SemanticRebuildStage::Both),
            skipped_reason: None,
        });
    }

    let mut recovered = 0usize;
    let mut skipped_packages = 0usize;
    let mut retries = 0usize;
    let mut last_cursor = None;

    for record in skipped {
        if let Some(package) = read_indexed_resource_package(state, &record.resource_did).await? {
            let cursor = record.cursor.max(1);
            let projected = DiscoveryProjectedPackage {
                cursor,
                package: &package,
                package_json: serde_json::to_value(&package).unwrap_or_else(|_| json!({})),
                projection: discovery_package_projection(&package),
            };
            match upsert_semantic_index_batch(state, &[projected], stage).await {
                Ok(()) => {
                    recovered += 1;
                    last_cursor = Some(cursor);
                    clear_semantic_rebuild_skip_record(state, stage, &record.resource_did).await?;
                }
                Err(err) => {
                    skipped_packages += 1;
                    retries += 1;
                    store_semantic_rebuild_skip(
                        state,
                        stage,
                        cursor,
                        record.resource_did.clone(),
                        err.to_string(),
                    )
                    .await?;
                }
            }
        }
    }

    Ok(SemanticRebuildReport {
        stage: stage.as_str().to_owned(),
        phase: "backfilled".to_owned(),
        backend: discovery_backend_name(state),
        semantic_enabled: true,
        package_count: recovered,
        context_indexed_count: count_semantic_indexed_resources(state)
            .await?
            .unwrap_or_default() as usize,
        intent_indexed_count: count_intent_indexed_rows(state).await?.unwrap_or_default() as usize,
        skipped_packages,
        recovered_packages: recovered,
        retries,
        last_cursor,
        safe_to_delete_old_indexes: skipped_packages == 0
            && matches!(stage, SemanticRebuildStage::Both),
        skipped_reason: None,
    })
}

async fn clear_semantic_rebuild_progress(
    state: &AppState,
    stage: SemanticRebuildStage,
) -> Result<()> {
    clear_discovery_sync_state_like(
        state,
        &format!("semantic_rebuild_progress:{}%", stage.as_str()),
    )
    .await
}

async fn clear_semantic_rebuild_skip_records(
    state: &AppState,
    stage: Option<SemanticRebuildStage>,
) -> Result<()> {
    clear_discovery_sync_state_like(state, &format!("{}%", semantic_rebuild_skip_prefix(stage)))
        .await
}

#[allow(clippy::too_many_arguments)]
async fn store_semantic_rebuild_progress(
    state: &AppState,
    stage: SemanticRebuildStage,
    phase: &str,
    last_cursor: Option<i64>,
    last_resource_did: Option<String>,
    processed: usize,
    total: usize,
    skipped_packages: usize,
    retries: usize,
    batch_size: usize,
) -> Result<()> {
    let record = SemanticRebuildProgressRecord {
        stage: stage.as_str().to_owned(),
        phase: phase.to_owned(),
        last_cursor,
        last_resource_did,
        processed,
        total,
        skipped_packages,
        retries,
        batch_size,
        updated_at: Utc::now().to_rfc3339(),
    };
    upsert_discovery_sync_state(state, &semantic_rebuild_progress_key(stage), &record).await
}

async fn store_semantic_rebuild_skip(
    state: &AppState,
    stage: SemanticRebuildStage,
    cursor: i64,
    resource_did: String,
    reason: String,
) -> Result<()> {
    let record = SemanticRebuildSkipRecord {
        stage: stage.as_str().to_owned(),
        resource_did,
        cursor,
        reason,
        updated_at: Utc::now().to_rfc3339(),
    };
    upsert_discovery_sync_state(
        state,
        &format!("semantic_rebuild_skip:{}:{cursor}", stage.as_str()),
        &record,
    )
    .await
}

async fn clear_semantic_rebuild_skip_record(
    state: &AppState,
    stage: SemanticRebuildStage,
    resource_did: &str,
) -> Result<()> {
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE ? AND json_extract(sync_value, '$.resourceDid') = ?"
        ))
        .bind(format!("semantic_rebuild_skip:{}:%", stage.as_str()))
        .bind(resource_did)
        .execute(sqlite.pool())
        .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE $1 AND sync_value->>'resourceDid' = $2"
        ))
        .bind(format!("semantic_rebuild_skip:{}:%", stage.as_str()))
        .bind(resource_did)
        .execute(postgres.pool())
        .await?;
    }
    Ok(())
}

async fn load_semantic_rebuild_skip_records(
    state: &AppState,
    stage: Option<SemanticRebuildStage>,
) -> Result<Vec<SemanticRebuildSkipRecord>> {
    let prefix = semantic_rebuild_skip_prefix(stage);
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query(&format!(
            "SELECT sync_value FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE ? ORDER BY sync_key"
        ))
        .bind(format!("{prefix}%"))
        .fetch_all(sqlite.pool())
        .await?;
        return Ok(rows
            .into_iter()
            .filter_map(|row| {
                serde_json::from_str::<SemanticRebuildSkipRecord>(&row.get::<String, _>(0)).ok()
            })
            .collect());
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            "SELECT sync_value::text FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE $1 ORDER BY sync_key"
        ))
        .bind(format!("{prefix}%"))
        .fetch_all(postgres.pool())
        .await?;
        return Ok(rows
            .into_iter()
            .filter_map(|row| {
                serde_json::from_str::<SemanticRebuildSkipRecord>(&row.get::<String, _>(0)).ok()
            })
            .collect());
    }
    Ok(vec![])
}

async fn clear_discovery_sync_state_like(state: &AppState, pattern: &str) -> Result<()> {
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE ?"
        ))
        .bind(pattern)
        .execute(sqlite.pool())
        .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!(
            "DELETE FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key LIKE $1"
        ))
        .bind(pattern)
        .execute(postgres.pool())
        .await?;
        return Ok(());
    }
    Ok(())
}

async fn load_semantic_rebuild_progress(
    state: &AppState,
    stage: SemanticRebuildStage,
) -> Result<Option<SemanticRebuildProgressRecord>> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!(
            "SELECT sync_value FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key = ?"
        ))
        .bind(semantic_rebuild_progress_key(stage))
        .fetch_optional(sqlite.pool())
        .await?;
        if let Some(row) = row {
            let value = row.get::<String, _>(0);
            return Ok(serde_json::from_str::<SemanticRebuildProgressRecord>(&value).ok());
        }
        return Ok(None);
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!(
            "SELECT sync_value::text FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE sync_key = $1"
        ))
        .bind(semantic_rebuild_progress_key(stage))
        .fetch_optional(postgres.pool())
        .await?;
        if let Some(row) = row {
            let value = row.get::<String, _>(0);
            return Ok(serde_json::from_str::<SemanticRebuildProgressRecord>(&value).ok());
        }
        return Ok(None);
    }
    Ok(None)
}

async fn upsert_discovery_sync_state<T: Serialize>(
    state: &AppState,
    key: &str,
    value: &T,
) -> Result<()> {
    let payload = serde_json::to_value(value)?;
    let updated_at = Utc::now();
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_SYNC_STATE_TABLE}(state_key, state_value, sync_key, sync_value, updated_at)
            VALUES (?1, ?2, ?1, ?2, ?3)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                sync_key = excluded.sync_key,
                sync_value = excluded.sync_value,
                updated_at = excluded.updated_at
            "#
        ))
        .bind(key)
        .bind(payload.to_string())
        .bind(updated_at)
        .execute(sqlite.pool())
        .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_SYNC_STATE_TABLE}(state_key, state_value, sync_key, sync_value, updated_at)
            VALUES ($1, $2::jsonb, $1, $2::jsonb, $3)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                sync_key = excluded.sync_key,
                sync_value = excluded.sync_value,
                updated_at = excluded.updated_at
            "#
        ))
        .bind(key)
        .bind(payload.to_string())
        .bind(updated_at)
        .execute(postgres.pool())
        .await?;
        return Ok(());
    }
    Ok(())
}

fn is_embedding_timeout_error(err: &anyhow::Error) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("timed out") || text.contains("timeout")
}

fn discovery_package_projection(package: &ResourcePackage) -> DiscoveryPackageProjection {
    let mut protocols = package
        .metadata
        .protocol_bindings
        .iter()
        .filter_map(|binding| binding["protocol"].as_str())
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    protocols.extend(
        package
            .metadata
            .services
            .iter()
            .filter_map(|service| service.protocol.as_ref())
            .map(|value| value.to_ascii_lowercase()),
    );
    protocols.sort();
    protocols.dedup();

    let mut service_endpoints = package
        .metadata
        .services
        .iter()
        .map(|service| service.service_endpoint.clone())
        .collect::<Vec<_>>();
    service_endpoints.sort();
    service_endpoints.dedup();

    let search_text = semantic_search_text(&[
        package.resource_did.clone(),
        package.resource_type.as_str().to_owned(),
        package.metadata.name.clone(),
        package.metadata.description.clone(),
        package.metadata.capability_tags.join(" "),
        protocols.join(" "),
        service_endpoints.join(" "),
    ]);
    let tag_text = semantic_search_text(&[
        package.resource_did.clone(),
        package.resource_type.as_str().to_owned(),
        package.metadata.capability_tags.join(" "),
        protocols.join(" "),
        service_endpoints.join(" "),
    ]);

    DiscoveryPackageProjection {
        resource_type: package.resource_type.as_str().to_owned(),
        lifecycle_state: package.metadata.lifecycle_state.clone(),
        capability_tags: package.metadata.capability_tags.clone(),
        protocols,
        service_endpoints,
        tag_text,
        search_text,
    }
}

async fn count_indexed_resource_packages(state: &AppState) -> Result<usize> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!("SELECT COUNT(*) FROM {DISCOVERY_PACKAGE_TABLE}"))
            .fetch_one(sqlite.pool())
            .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!("SELECT COUNT(*) FROM {DISCOVERY_PACKAGE_TABLE}"))
            .fetch_one(postgres.pool())
            .await?;
        return Ok(row.get::<i64, _>(0) as usize);
    }
    Ok(state
        .index
        .read::<Vec<ResourcePackage>>("resource-capabilities.json")
        .unwrap_or_default()
        .len())
}

async fn read_indexed_resource_package(
    state: &AppState,
    did: &str,
) -> Result<Option<ResourcePackage>> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!(
            "SELECT package_json FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did = ?"
        ))
        .bind(did)
        .fetch_optional(sqlite.pool())
        .await?;
        return row
            .map(|row| {
                serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0))
                    .map_err(anyhow::Error::from)
            })
            .transpose();
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!(
            "SELECT package_json::text FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did = $1"
        ))
        .bind(did)
        .fetch_optional(postgres.pool())
        .await?;
        return row
            .map(|row| {
                serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0))
                    .map_err(anyhow::Error::from)
            })
            .transpose();
    }
    Ok(state
        .index
        .read::<Vec<ResourcePackage>>("resource-capabilities.json")
        .unwrap_or_default()
        .into_iter()
        .find(|package| package.resource_did == did))
}

async fn read_indexed_resource_cursor(state: &AppState, did: &str) -> Result<Option<i64>> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!(
            "SELECT cursor FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did = ?"
        ))
        .bind(did)
        .fetch_optional(sqlite.pool())
        .await?;
        return Ok(row.map(|row| row.get::<i64, _>(0)));
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!(
            "SELECT cursor FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did = $1"
        ))
        .bind(did)
        .fetch_optional(postgres.pool())
        .await?;
        return Ok(row.map(|row| row.get::<i64, _>(0)));
    }
    Ok(read_indexed_resource_package(state, did).await?.map(|_| 0))
}

async fn indexed_resource_visibility(state: &AppState, dids: &[String]) -> Result<Vec<String>> {
    if dids.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(sqlite) = &state.sqlite {
        let mut visible = Vec::new();
        for chunk in dids.chunks(500) {
            let mut builder = QueryBuilder::<Sqlite>::new(format!(
                "SELECT resource_did FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did IN ("
            ));
            let mut separated = builder.separated(", ");
            for did in chunk {
                separated.push_bind(did);
            }
            separated.push_unseparated(")");
            let rows = builder.build().fetch_all(sqlite.pool()).await?;
            visible.extend(rows.into_iter().map(|row| row.get::<String, _>(0)));
        }
        return Ok(visible);
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            "SELECT resource_did FROM {DISCOVERY_PACKAGE_TABLE} WHERE resource_did = ANY($1)"
        ))
        .bind(dids)
        .fetch_all(postgres.pool())
        .await?;
        return Ok(rows
            .into_iter()
            .map(|row| row.get::<String, _>(0))
            .collect());
    }
    let visible = state
        .index
        .read::<Vec<ResourcePackage>>("resource-capabilities.json")
        .unwrap_or_default()
        .into_iter()
        .filter_map(|package| {
            dids.iter()
                .any(|did| did == &package.resource_did)
                .then_some(package.resource_did)
        })
        .collect();
    Ok(visible)
}

async fn query_indexed_resource_packages(
    state: &AppState,
    query: &ResourceDiscoveryQuery,
) -> Result<(Vec<ResourcePackage>, bool)> {
    if let Some(postgres) = &state.postgres {
        let mut builder: QueryBuilder<Postgres> = QueryBuilder::new(format!(
            "SELECT package_json::text FROM {DISCOVERY_PACKAGE_TABLE} WHERE lifecycle_state = "
        ));
        builder.push_bind("active");
        if let Some(resource_type) = &query.resource_type {
            builder.push(" AND resource_type = ");
            builder.push_bind(resource_type.as_str());
        } else {
            builder.push(" AND resource_type = ANY(");
            builder.push_bind(default_discoverable_resource_type_names());
            builder.push(")");
        }
        if let Some(version) = &query.version {
            builder.push(" AND version = ");
            builder.push_bind(version);
        }
        if !query.capability_tags.is_empty() {
            builder.push(" AND capability_tags @> ");
            builder.push_bind(query.capability_tags.clone());
        }
        if let Some(protocol) = &query.protocol {
            builder.push(" AND protocols && ");
            builder.push_bind(vec![protocol.to_ascii_lowercase()]);
        }
        let tag_tokens = semantic_query_tag_tokens(query);
        let query_text = query
            .query
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty());
        if let Some(text) = query_text {
            builder.push(" AND (");
            builder.push("to_tsvector('simple', tag_text) @@ plainto_tsquery('simple', ");
            builder.push_bind(text.to_ascii_lowercase());
            builder.push(") OR to_tsvector('simple', search_text) @@ plainto_tsquery('simple', ");
            builder.push_bind(text.to_ascii_lowercase());
            builder.push("))");
        }
        if !tag_tokens.is_empty() {
            builder.push(" ORDER BY ");
            builder.push("ts_rank_cd(to_tsvector('simple', tag_text), plainto_tsquery('simple', ");
            if let Some(text) = query_text {
                builder.push_bind(text.to_ascii_lowercase());
            } else {
                builder.push_bind(tag_tokens.join(" "));
            }
            builder.push(")) DESC, ");
            builder
                .push("ts_rank_cd(to_tsvector('simple', search_text), plainto_tsquery('simple', ");
            if let Some(text) = query_text {
                builder.push_bind(text.to_ascii_lowercase());
            } else {
                builder.push_bind(tag_tokens.join(" "));
            }
            builder.push(")) DESC, updated_at DESC, resource_did LIMIT ");
        } else if should_use_sql_text_prefilter(query) {
            let Some(text) = query
                .query
                .as_ref()
                .filter(|value| !value.trim().is_empty())
            else {
                unreachable!("text prefilter requires a non-empty query");
            };
            builder.push(
                " ORDER BY ts_rank_cd(to_tsvector('simple', search_text), plainto_tsquery('simple', ",
            );
            builder.push_bind(text.to_ascii_lowercase());
            builder.push(")) DESC, updated_at DESC, resource_did LIMIT ");
        } else {
            builder.push(" ORDER BY updated_at DESC, resource_did LIMIT ");
        }
        builder.push_bind(discovery_sql_candidate_limit_for_query(query));
        let rows = builder.build().fetch_all(postgres.pool()).await?;
        let packages = rows
            .into_iter()
            .map(|row| serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok((filter_packages_for_local_discovery(state, packages)?, true));
    }
    read_indexed_resource_packages(state)
        .await
        .and_then(|packages| {
            let packages = packages
                .into_iter()
                .filter(|package| should_search_resource_type(package, query))
                .collect::<Vec<_>>();
            Ok((filter_packages_for_local_discovery(state, packages)?, false))
        })
}

async fn query_semantic_index(
    state: &AppState,
    query: &ResourceDiscoveryQuery,
) -> Result<Option<Vec<SemanticQueryHit>>> {
    if !semantic_search_available(state) {
        return Ok(None);
    }
    let engine = DiscoverySearchEngine::from_config(&state.config.semantic_search)?;
    engine.query(state, query).await
}

async fn query_pgvector_semantic_index(
    state: &AppState,
    query: &ResourceDiscoveryQuery,
) -> Result<Option<Vec<SemanticQueryHit>>> {
    if !semantic_search_available(state) {
        return Ok(None);
    }
    let Some(text) = query
        .query
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let Some(postgres) = &state.postgres else {
        return Ok(None);
    };

    let query_embedding =
        embed_semantic_text(&state.config.semantic_search, &state.client, text).await?;
    let candidate_limit = semantic_candidate_limit(query.limit, &state.config.semantic_search);
    let mut builder = QueryBuilder::<Postgres>::new(format!(
        r#"
        SELECT
            p.package_json::text,
            (1 - (s.embedding <=> {}::vector))::float4 AS semantic_score
        FROM {DISCOVERY_SEMANTIC_INDEX_TABLE} s
        JOIN {DISCOVERY_PACKAGE_TABLE} p ON p.resource_did = s.resource_did
        WHERE p.lifecycle_state = 
        "#,
        pgvector_literal(&query_embedding),
    ));
    builder.push_bind("active");
    builder.push(" AND s.lifecycle_state = ");
    builder.push_bind("active");
    builder.push(" AND s.embedding_model = ");
    builder.push_bind(&state.semantic.embedding_model);
    builder.push(" AND s.embedding_version = ");
    builder.push_bind(&state.semantic.embedding_version);

    if let Some(resource_type) = &query.resource_type {
        builder.push(" AND p.resource_type = ");
        builder.push_bind(resource_type.as_str());
    } else {
        builder.push(" AND p.resource_type = ANY(");
        builder.push_bind(default_discoverable_resource_type_names());
        builder.push(")");
    }
    if let Some(version) = &query.version {
        builder.push(" AND p.version = ");
        builder.push_bind(version);
    }
    if !query.capability_tags.is_empty()
        && state
            .config
            .semantic_search
            .filters
            .capability_tags
            .eq_ignore_ascii_case("hard")
    {
        builder.push(" AND p.capability_tags @> ");
        builder.push_bind(query.capability_tags.clone());
    }
    if let Some(protocol) = &query.protocol {
        if state
            .config
            .semantic_search
            .filters
            .protocol
            .eq_ignore_ascii_case("hard")
        {
            builder.push(" AND p.protocols && ");
            builder.push_bind(vec![protocol.to_ascii_lowercase()]);
        }
    }
    builder.push(format!(
        " ORDER BY s.embedding <=> {}::vector LIMIT ",
        pgvector_literal(&query_embedding)
    ));
    builder.push_bind(candidate_limit);

    let rows = builder.build().fetch_all(postgres.pool()).await?;
    let vector_hits = rows
        .into_iter()
        .map(|row| {
            let package = serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0))?;
            let vector_score = row.get::<f32, _>(1).clamp(0.0, 1.0);
            Ok::<_, anyhow::Error>((package, vector_score))
        })
        .collect::<Result<Vec<_>>>()?;
    let (lexical_packages, _) = query_indexed_resource_packages(state, query).await?;
    let mut hits = hybrid_semantic_hits(vector_hits, lexical_packages, query);
    if let Err(err) = apply_pgvector_intent_maxsim(state, &query_embedding, &mut hits).await {
        eprintln!("intent maxsim rerank skipped: {err}");
    }
    let allowed_domains = local_discovery_authorized_domains(state)?;
    hits.retain(|hit| {
        authorized_domains_cover(&allowed_domains, &hit.package.metadata.authorized_domains)
    });
    hits.sort_by(|a, b| b.final_score.total_cmp(&a.final_score));
    hits.truncate(query.limit as usize);
    Ok(Some(hits))
}

async fn apply_pgvector_intent_maxsim(
    state: &AppState,
    query_embedding: &[f32],
    hits: &mut [SemanticQueryHit],
) -> Result<()> {
    let Some(postgres) = &state.postgres else {
        return Ok(());
    };
    if hits.is_empty() {
        return Ok(());
    }
    let dids = hits
        .iter()
        .map(|hit| hit.package.resource_did.clone())
        .collect::<Vec<_>>();
    let rows = sqlx::query(&format!(
        r#"
        SELECT resource_did, intent_text, intent_score
        FROM (
            SELECT
                resource_did,
                intent_text,
                (1 - (embedding <=> {}::vector))::float4 AS intent_score,
                ROW_NUMBER() OVER (
                    PARTITION BY resource_did
                    ORDER BY embedding <=> {}::vector
                ) AS rank
            FROM {DISCOVERY_INTENT_INDEX_TABLE}
            WHERE resource_did = ANY($1)
              AND lifecycle_state = 'active'
              AND embedding_model = $2
              AND embedding_version = $3
        ) ranked
        WHERE rank = 1
        "#,
        pgvector_literal(query_embedding),
        pgvector_literal(query_embedding),
    ))
    .bind(&dids)
    .bind(&state.semantic.embedding_model)
    .bind(&state.semantic.embedding_version)
    .fetch_all(postgres.pool())
    .await?;

    let maxsim_by_did = rows
        .into_iter()
        .map(|row| {
            (
                row.get::<String, _>("resource_did"),
                (
                    row.get::<f32, _>("intent_score").clamp(0.0, 1.0),
                    row.get::<String, _>("intent_text"),
                ),
            )
        })
        .collect::<HashMap<_, _>>();

    for hit in hits {
        if let Some((intent_score, intent_text)) = maxsim_by_did.get(&hit.package.resource_did) {
            if *intent_score > hit.intent_score {
                hit.intent_score = *intent_score;
                hit.matched_intent = Some(intent_text.clone());
                hit.semantic_score =
                    (0.80 * hit.semantic_score + 0.20 * intent_score).clamp(0.0, 1.0);
                hit.final_score =
                    (hit.semantic_score + hit.structured_score + 0.03).clamp(0.0, 1.0);
            }
        }
    }
    Ok(())
}

fn hybrid_semantic_hits(
    vector_hits: Vec<(ResourcePackage, f32)>,
    lexical_packages: Vec<ResourcePackage>,
    query: &ResourceDiscoveryQuery,
) -> Vec<SemanticQueryHit> {
    let query_projection = semantic_query_projection(query);
    let mut vector_scores = HashMap::new();
    let mut packages = Vec::new();
    let mut seen = HashSet::new();
    for (package, vector_score) in vector_hits {
        vector_scores.insert(package.resource_did.clone(), vector_score.clamp(0.0, 1.0));
        if seen.insert(package.resource_did.clone()) {
            packages.push(package);
        }
    }
    for package in lexical_packages {
        if seen.insert(package.resource_did.clone()) {
            packages.push(package);
        }
    }
    let doc_tokens = packages
        .iter()
        .map(|package| {
            let projection = discovery_package_projection(package);
            let doc = search_document_from_package(0, package, &projection);
            semantic_tokens(&semantic_context_text(&doc))
        })
        .collect::<Vec<_>>();
    let mut hits = packages
        .into_iter()
        .enumerate()
        .map(|(index, package)| {
            let vector_score = vector_scores
                .get(&package.resource_did)
                .copied()
                .unwrap_or(0.0);
            let bm25_score = bm25_like_score(&query_projection.expanded_tokens, &doc_tokens, index);
            let mut hit = semantic_resource_scores(&package, query, &query_projection, bm25_score);
            hit.semantic_score = (0.45 * vector_score) + (0.55 * hit.semantic_score);
            hit.final_score = (hit.semantic_score + hit.structured_score + 0.03).clamp(0.0, 1.0);
            hit
        })
        .filter(|hit| resource_matches_explicit_constraints(&hit.package, query))
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| b.final_score.total_cmp(&a.final_score));
    hits.truncate(query.limit as usize);
    hits
}

fn filter_packages_for_local_discovery(
    state: &AppState,
    packages: Vec<ResourcePackage>,
) -> Result<Vec<ResourcePackage>> {
    let allowed_domains = local_discovery_authorized_domains(state)?;
    Ok(packages
        .into_iter()
        .filter(|package| {
            authorized_domains_cover(&allowed_domains, &package.metadata.authorized_domains)
        })
        .collect())
}

fn local_discovery_authorized_domains(state: &AppState) -> Result<Vec<String>> {
    let document: DidDocument = state.data.read("did-document.json")?;
    let domains = document
        .oan_metadata
        .as_ref()
        .map(|metadata| metadata.authorized_domains.clone())
        .unwrap_or_default();
    validate_authorized_domain_list(&domains).map_err(anyhow::Error::msg)?;
    Ok(domains)
}

fn validate_authorized_domain_list(domains: &[String]) -> std::result::Result<(), String> {
    if domains.iter().any(|domain| domain == "*") {
        return if domains.len() == 1 {
            Ok(())
        } else {
            Err("invalid_authorized_domains".to_owned())
        };
    }
    for domain in domains {
        if domain.trim().is_empty()
            || domain != domain.trim()
            || domain.contains("..")
            || domain.starts_with('.')
            || domain.ends_with('.')
        {
            return Err("invalid_authorized_domains".to_owned());
        }
    }
    if domains.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("invalid_authorized_domains".to_owned());
    }
    Ok(())
}

fn authorized_domain_covers(granted: &str, requested: &str) -> bool {
    granted == requested
        || requested
            .strip_prefix(granted)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

fn authorized_domains_cover(granted: &[String], requested: &[String]) -> bool {
    if requested.is_empty() {
        return false;
    }
    if granted.iter().any(|domain| domain == "*") {
        return true;
    }
    if granted.is_empty() {
        return false;
    }
    requested.iter().all(|requested_domain| {
        granted
            .iter()
            .any(|granted_domain| authorized_domain_covers(granted_domain, requested_domain))
    })
}

fn semantic_candidate_limit(query_limit: u32, config: &SemanticSearchConfig) -> i64 {
    let requested = query_limit.max(1);
    let expanded = requested.saturating_mul(config.candidate_multiplier.max(1));
    expanded
        .max(25)
        .min(config.max_candidates.max(requested))
        .max(requested) as i64
}

fn discovery_sql_candidate_limit(query_limit: u32) -> i64 {
    let requested = query_limit.max(1) as i64;
    (requested * 3).clamp(25, 5_000).max(requested)
}

fn discovery_sql_candidate_limit_for_query(query: &ResourceDiscoveryQuery) -> i64 {
    if should_use_sql_text_prefilter(query) {
        return discovery_sql_candidate_limit(query.limit);
    }
    let requested = query.limit.max(1) as i64;
    (requested * 50).clamp(200, 5_000).max(requested)
}

fn validated_discovery_page_limit(limit: Option<u32>) -> std::result::Result<u32, ApiError> {
    let limit = limit.unwrap_or(DISCOVERY_DEFAULT_PAGE_SIZE);
    if limit == 0 || limit > DISCOVERY_MAX_PAGE_SIZE {
        return Err(ApiError::bad_request("invalid_discovery_page_limit"));
    }
    Ok(limit)
}

fn resource_package_summary(cursor: i64, package: &ResourcePackage) -> Value {
    json!({
        "cursor": cursor,
        "resourceDid": package.resource_did,
        "resourceType": package.resource_type,
        "name": package.metadata.name,
        "description": package.metadata.description,
        "capabilityTags": package.metadata.capability_tags,
        "services": package.metadata.services,
        "lifecycleState": package.metadata.lifecycle_state,
        "version": package.package_version,
        "updatedAt": package.metadata.updated_at
    })
}

async fn read_indexed_resource_page(
    state: &AppState,
    after_cursor: i64,
    after_resource_did: Option<&str>,
    limit: u32,
) -> Result<CursorPage<(i64, ResourcePackage)>> {
    let page_limit = i64::from(limit.max(1));
    let mut items = Vec::new();
    let mut page_bytes = 0_usize;
    let has_more;
    if let Some(sqlite) = &state.sqlite {
        if let Some(after_resource_did) = after_resource_did {
            let query = format!(
                r#"
                SELECT cursor, package_json
                FROM {DISCOVERY_PACKAGE_TABLE}
                WHERE cursor > ? OR (cursor = ? AND resource_did > ?)
                ORDER BY cursor, resource_did
                LIMIT ?
                "#
            );
            let mut rows = sqlx::query(&query)
                .bind(after_cursor)
                .bind(after_cursor)
                .bind(after_resource_did)
                .bind(page_limit)
                .fetch(sqlite.pool());
            while let Some(row) = rows.try_next().await? {
                push_indexed_resource_page_item(
                    &mut items,
                    &mut page_bytes,
                    row.get::<i64, _>(0),
                    &row.get::<String, _>(1),
                )?;
            }
            has_more = indexed_resource_page_has_more_sqlite(
                state,
                items
                    .last()
                    .map(|(cursor, package)| (*cursor, package.resource_did.as_str()))
                    .or(Some((after_cursor, after_resource_did))),
            )
            .await?;
        } else {
            let query = format!(
                r#"
                SELECT cursor, package_json
                FROM {DISCOVERY_PACKAGE_TABLE}
                WHERE cursor > ?
                ORDER BY cursor, resource_did
                LIMIT ?
                "#
            );
            let mut rows = sqlx::query(&query)
                .bind(after_cursor)
                .bind(page_limit)
                .fetch(sqlite.pool());
            while let Some(row) = rows.try_next().await? {
                push_indexed_resource_page_item(
                    &mut items,
                    &mut page_bytes,
                    row.get::<i64, _>(0),
                    &row.get::<String, _>(1),
                )?;
            }
            has_more = indexed_resource_page_has_more_sqlite(
                state,
                items
                    .last()
                    .map(|(cursor, package)| (*cursor, package.resource_did.as_str())),
            )
            .await?;
        }
    } else if let Some(postgres) = &state.postgres {
        if let Some(after_resource_did) = after_resource_did {
            let query = format!(
                r#"
                SELECT cursor, package_json::text
                FROM {DISCOVERY_PACKAGE_TABLE}
                WHERE cursor > $1 OR (cursor = $1 AND resource_did > $2)
                ORDER BY cursor, resource_did
                LIMIT $3
                "#
            );
            let mut rows = sqlx::query(&query)
                .bind(after_cursor)
                .bind(after_resource_did)
                .bind(page_limit)
                .fetch(postgres.pool());
            while let Some(row) = rows.try_next().await? {
                push_indexed_resource_page_item(
                    &mut items,
                    &mut page_bytes,
                    row.get::<i64, _>(0),
                    &row.get::<String, _>(1),
                )?;
            }
            has_more = indexed_resource_page_has_more_postgres(
                state,
                items
                    .last()
                    .map(|(cursor, package)| (*cursor, package.resource_did.as_str()))
                    .or(Some((after_cursor, after_resource_did))),
            )
            .await?;
        } else {
            let query = format!(
                r#"
                SELECT cursor, package_json::text
                FROM {DISCOVERY_PACKAGE_TABLE}
                WHERE cursor > $1
                ORDER BY cursor, resource_did
                LIMIT $2
                "#
            );
            let mut rows = sqlx::query(&query)
                .bind(after_cursor)
                .bind(page_limit)
                .fetch(postgres.pool());
            while let Some(row) = rows.try_next().await? {
                push_indexed_resource_page_item(
                    &mut items,
                    &mut page_bytes,
                    row.get::<i64, _>(0),
                    &row.get::<String, _>(1),
                )?;
            }
            has_more = indexed_resource_page_has_more_postgres(
                state,
                items
                    .last()
                    .map(|(cursor, package)| (*cursor, package.resource_did.as_str())),
            )
            .await?;
        }
    } else {
        let mut indexed = state
            .index
            .read::<Vec<ResourcePackage>>("resource-capabilities.json")
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, package)| (index as i64 + 1, package))
            .filter(|(cursor, package)| {
                *cursor > after_cursor
                    || (*cursor == after_cursor
                        && after_resource_did
                            .map(|did| package.resource_did.as_str() > did)
                            .unwrap_or(false))
            })
            .collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.resource_did.cmp(&right.1.resource_did))
        });
        has_more = indexed.len() > page_limit as usize;
        items = indexed.into_iter().take(page_limit as usize).collect();
        let mut bounded_items = Vec::with_capacity(items.len());
        for (cursor, package) in items {
            let package_json = serde_json::to_vec(&package)?;
            if package_json.len() > DISCOVERY_INDEX_PACKAGE_BYTES {
                return Err(anyhow!("discovery_index_package_too_large"));
            }
            page_bytes = page_bytes.saturating_add(package_json.len());
            if page_bytes > DISCOVERY_INDEX_PAGE_BYTES {
                return Err(anyhow!("discovery_index_page_too_large"));
            }
            bounded_items.push((cursor, package));
        }
        items = bounded_items;
    }
    let next_cursor = items
        .last()
        .map(|(cursor, _)| *cursor)
        .unwrap_or(after_cursor);
    let next_resource_did = items
        .last()
        .map(|(_, package)| package.resource_did.clone());
    Ok(CursorPage {
        items,
        next_cursor,
        next_resource_did,
        has_more,
    })
}

async fn read_indexed_resource_summary_page(
    state: &AppState,
    after_cursor: i64,
    after_resource_did: Option<&str>,
    limit: u32,
) -> Result<CursorPage<Value>> {
    let page = read_indexed_resource_page(state, after_cursor, after_resource_did, limit).await?;
    Ok(CursorPage {
        next_cursor: page.next_cursor,
        next_resource_did: page.next_resource_did,
        has_more: page.has_more,
        items: page
            .items
            .into_iter()
            .map(|(cursor, package)| resource_package_summary(cursor, &package))
            .collect(),
    })
}

async fn indexed_resource_page_has_more_sqlite(
    state: &AppState,
    position: Option<(i64, &str)>,
) -> Result<bool> {
    let Some(sqlite) = &state.sqlite else {
        return Ok(false);
    };
    let Some((cursor, resource_did)) = position else {
        return Ok(false);
    };
    Ok(sqlx::query(&format!(
        "SELECT 1 FROM {DISCOVERY_PACKAGE_TABLE}
         WHERE cursor > ? OR (cursor = ? AND resource_did > ?)
         LIMIT 1"
    ))
    .bind(cursor)
    .bind(cursor)
    .bind(resource_did)
    .fetch_optional(sqlite.pool())
    .await?
    .is_some())
}

async fn indexed_resource_page_has_more_postgres(
    state: &AppState,
    position: Option<(i64, &str)>,
) -> Result<bool> {
    let Some(postgres) = &state.postgres else {
        return Ok(false);
    };
    let Some((cursor, resource_did)) = position else {
        return Ok(false);
    };
    Ok(sqlx::query(&format!(
        "SELECT 1 FROM {DISCOVERY_PACKAGE_TABLE}
         WHERE cursor > $1 OR (cursor = $1 AND resource_did > $2)
         LIMIT 1"
    ))
    .bind(cursor)
    .bind(resource_did)
    .fetch_optional(postgres.pool())
    .await?
    .is_some())
}

fn push_indexed_resource_page_item(
    items: &mut Vec<(i64, ResourcePackage)>,
    page_bytes: &mut usize,
    cursor: i64,
    package_json: &str,
) -> Result<()> {
    let package_bytes = package_json.len();
    if package_bytes > DISCOVERY_INDEX_PACKAGE_BYTES {
        return Err(anyhow!("discovery_index_package_too_large"));
    }
    *page_bytes = page_bytes.saturating_add(package_bytes);
    if *page_bytes > DISCOVERY_INDEX_PAGE_BYTES {
        return Err(anyhow!("discovery_index_page_too_large"));
    }
    items.push((
        cursor,
        serde_json::from_str::<ResourcePackage>(package_json)?,
    ));
    Ok(())
}

async fn read_visible_indexed_resource_did_page(
    state: &AppState,
    after_cursor: i64,
    after_resource_did: Option<&str>,
    limit: u32,
) -> Result<CursorPage<String>> {
    let page =
        read_visible_indexed_resource_page(state, after_cursor, after_resource_did, limit).await?;
    let next_cursor = page.next_cursor;
    let has_more = page.has_more;
    Ok(CursorPage {
        items: page
            .items
            .into_iter()
            .map(|(_, package)| package.resource_did)
            .collect(),
        has_more,
        next_cursor,
        next_resource_did: page.next_resource_did,
    })
}

async fn read_indexed_resource_packages(state: &AppState) -> Result<Vec<ResourcePackage>> {
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query(&format!(
            "SELECT package_json FROM {DISCOVERY_PACKAGE_TABLE} ORDER BY updated_at, resource_did"
        ))
        .fetch_all(sqlite.pool())
        .await?;
        if !rows.is_empty() {
            let packages = rows
                .into_iter()
                .map(|row| serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(packages);
        }
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            "SELECT package_json::text FROM {DISCOVERY_PACKAGE_TABLE} ORDER BY updated_at, resource_did"
        ))
        .fetch_all(postgres.pool())
        .await?;
        if !rows.is_empty() {
            let packages = rows
                .into_iter()
                .map(|row| serde_json::from_str::<ResourcePackage>(&row.get::<String, _>(0)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(packages);
        }
    }
    Ok(state
        .index
        .read("resource-capabilities.json")
        .unwrap_or_default())
}

async fn write_indexed_resource_packages(
    state: &AppState,
    packages: &[ResourcePackage],
) -> Result<()> {
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!("DELETE FROM {DISCOVERY_PACKAGE_TABLE}"))
            .execute(sqlite.pool())
            .await?;
        let items = packages
            .iter()
            .enumerate()
            .map(|(cursor, package)| (cursor as i64 + 1, package.clone()))
            .collect::<Vec<_>>();
        upsert_indexed_resource_packages_batch(state, &items).await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!("DELETE FROM {DISCOVERY_PACKAGE_TABLE}"))
            .execute(postgres.pool())
            .await?;
        let items = packages
            .iter()
            .enumerate()
            .map(|(cursor, package)| (cursor as i64 + 1, package.clone()))
            .collect::<Vec<_>>();
        upsert_indexed_resource_packages_batch(state, &items).await?;
        return Ok(());
    }
    state
        .index
        .write("resource-capabilities.json", &packages.to_vec())?;
    Ok(())
}

async fn seed_semantic_evaluation_packages(state: &AppState, suite_path: &Path) -> Result<usize> {
    let suite: SemanticEvaluationSuite =
        serde_json::from_str(&read_utf8_text_allowing_bom(suite_path)?)?;
    let mut packages = suite.seed_packages;
    packages.extend(
        suite
            .seed_resources
            .iter()
            .map(seed_resource_package)
            .collect::<Result<Vec<_>>>()?,
    );
    if packages.is_empty() {
        return Ok(0);
    }
    for package in &packages {
        validate_resource_package_for_index(package)
            .map_err(|reason| anyhow!("invalid_seed_package:{}:{reason}", package.resource_did))?;
    }
    let count = packages.len();
    write_indexed_resource_packages(state, &packages).await?;
    Ok(count)
}

fn seed_resource_package(seed: &SemanticEvaluationSeedResource) -> Result<ResourcePackage> {
    let did = seed.resource_did.clone();
    let service_endpoint = seed.service_endpoint.clone().unwrap_or_else(|| {
        format!(
            "https://example.org/oan/resources/{}",
            did.replace(':', "/")
        )
    });
    let protocol = seed
        .protocols
        .first()
        .cloned()
        .unwrap_or_else(|| "https".to_owned());
    let service = ServiceEndpoint {
        id: format!("{did}#service-1"),
        service_type: format!("{}Service", seed.resource_type.as_str()),
        service_endpoint: service_endpoint.clone(),
        version: Some(seed.package_version.clone()),
        protocol: Some(protocol.clone()),
        server_type: None,
        port: None,
    };
    let binding = ProtocolBinding {
        id: format!("{did}#binding-{protocol}"),
        protocol: protocol.clone(),
        version: None,
        transport: Some(if protocol.eq_ignore_ascii_case("https") {
            "http".to_owned()
        } else {
            protocol.clone()
        }),
        service_ref: Some(service.id.clone()),
        schema_ref: None,
        extra: Default::default(),
    };
    let authorized_domains = if seed.authorized_domains.is_empty() {
        vec!["examples".to_owned()]
    } else {
        seed.authorized_domains.clone()
    };
    let description = ResourceDescription {
        name: Some(seed.name.clone()),
        description: Some(seed.description.clone()),
        capability_description: Some(seed.capability_description.clone()),
        capability_tags: seed.capability_tags.clone(),
        use_case_examples: seed.use_case_examples.clone(),
        examples: seed.examples.clone(),
        version: Some(seed.package_version.clone()),
        ..Default::default()
    };
    let key = generate_ed25519_keypair();
    let signing_key = SigningKey::Ed25519 {
        suite: CryptoSuite::Ed25519Sha256,
        key: key.clone(),
    };
    let verifying_key = VerifyingKey::Ed25519 {
        suite: CryptoSuite::Ed25519Sha256,
        key: key.verifying_key(),
    };
    let key_id = format!("{did}#key-1");
    let mut did_document = DidDocument {
        context: vec![
            "https://www.w3.org/ns/did/v1".to_owned(),
            "https://openagenet.xyz/did-oan-specs/v1".to_owned(),
            "https://w3id.org/security/suites/ed25519-2020/v1".to_owned(),
        ],
        id: did.clone(),
        controller: Some(oan_core::DidController::Did(did.clone())),
        verification_method: vec![VerificationMethod {
            id: key_id.clone(),
            method_type: "Ed25519VerificationKey2020".to_owned(),
            controller: did.clone(),
            crypto_suite: Some(CryptoSuite::Ed25519Sha256),
            public_key_format: None,
            public_key_multibase: Some(public_key_multibase(&verifying_key)),
            public_key_jwk: Some(public_key_jwk(&verifying_key)),
        }],
        authentication: vec![key_id.clone()],
        assertion_method: vec![key_id.clone()],
        capability_invocation: vec![key_id.clone()],
        service: vec![service.clone()],
        proof: None,
        oan_metadata: Some(OanMetadata {
            subject_type: match seed.resource_type {
                ResourceType::AgentService => SubjectType::AgentService,
                ResourceType::Skill => SubjectType::Skill,
                ResourceType::McpServer => SubjectType::McpServer,
                ResourceType::ToolApi => SubjectType::ToolApi,
                _ => SubjectType::Unspecified,
            },
            resource_type: seed.resource_type.clone(),
            external_identifiers: vec![],
            identity_type: None,
            controller_did: None,
            publisher_did: Some("did:oan:R4tYu:semantic-evaluation-publisher".to_owned()),
            issuer_did: None,
            ttl: None,
            resource_description: Some(description),
            agent_description: None,
            capability_tags: seed.capability_tags.clone(),
            authorized_domains: authorized_domains.clone(),
            protocol_bindings: vec![binding],
            implementation_links: vec![ImplementationLink {
                relation: "service".to_owned(),
                target_did: did.clone(),
                target_type: Some(seed.resource_type.clone()),
                target_service: Some(service.id.clone()),
                version_constraint: Some(seed.package_version.clone()),
            }],
            credential_requirements: vec![],
            package_info: None,
            service_policy: None,
            network_scope: Some("oan-evaluation".to_owned()),
            lifecycle_state: Some("active".to_owned()),
            extra: Default::default(),
        }),
    };
    let proof_input = did_document_signature_input(&did_document, CryptoSuite::Ed25519Sha256)?;
    did_document.proof = Some(oan_core::DataIntegrityProof {
        context: None,
        proof_type: "Ed25519Signature2020".to_owned(),
        creator: key_id.clone(),
        created: Utc::now(),
        proof_purpose: "assertionMethod".to_owned(),
        proof_value: sign_bytes_multibase(&signing_key, &proof_input)?,
        crypto_suite: None,
        hash_algorithm: None,
        verification_method: Some(key_id),
    });
    let mut package = ResourcePackage {
        package_version: seed.package_version.clone(),
        resource_did: did.clone(),
        resource_type: seed.resource_type.clone(),
        did_document,
        did_document_hash: String::new(),
        metadata_hash: String::new(),
        package_hash: String::new(),
        hash_algorithm: "sha256".to_owned(),
        metadata: ResourceMetadata {
            resource_did: did.clone(),
            resource_type: seed.resource_type.clone(),
            subject_type: match seed.resource_type {
                ResourceType::AgentService => SubjectType::AgentService,
                ResourceType::Skill => SubjectType::Skill,
                ResourceType::McpServer => SubjectType::McpServer,
                ResourceType::ToolApi => SubjectType::ToolApi,
                _ => SubjectType::Unspecified,
            },
            publisher_did: Some("did:oan:R4tYu:semantic-evaluation-publisher".to_owned()),
            subject_did: Some(did.clone()),
            name: seed.name.clone(),
            description: seed.description.clone(),
            capability_tags: seed.capability_tags.clone(),
            authorized_domains,
            protocol_bindings: vec![json!({"protocol": protocol, "role": "service"})],
            services: vec![service],
            lifecycle_state: "active".to_owned(),
            package_version: seed.package_version.clone(),
            package_hash: String::new(),
            metadata_hash: String::new(),
            hash_algorithm: "sha256".to_owned(),
            updated_at: Utc::now(),
        },
        root_proof: RootProof {
            root_did: "did:oan:QwErT:semantic-evaluation-root".to_owned(),
            bulletin_event_hash: None,
            signature: None,
            package_claims: None,
            proof: None,
            crypto_suite: Some(CryptoSuite::Ed25519Sha256),
            hash_algorithm: Some("sha256".to_owned()),
        },
        created_at: Utc::now(),
    };
    refresh_resource_package_hashes(&mut package)?;
    Ok(package)
}

fn refresh_resource_package_hashes(package: &mut ResourcePackage) -> Result<()> {
    package.did_document_hash =
        hash_json_with_suite(CryptoSuite::Ed25519Sha256, &package.did_document)
            .map(|hash| format!("sha256:{hash}"))?;
    package.metadata.metadata_hash.clear();
    package.metadata.package_hash.clear();
    package.metadata_hash =
        hash_resource_metadata_with_suite(CryptoSuite::Ed25519Sha256, &package.metadata)
            .map(|hash| format!("sha256:{hash}"))?;
    package.metadata.metadata_hash = package.metadata_hash.clone();
    package.package_hash = hash_json_with_suite(
        CryptoSuite::Ed25519Sha256,
        &json!({
            "packageVersion": package.package_version,
            "resourceDid": package.resource_did,
            "resourceType": package.resource_type,
            "didDocumentHash": package.did_document_hash,
            "metadataHash": package.metadata_hash,
            "hashAlgorithm": package.hash_algorithm,
        }),
    )
    .map(|hash| format!("sha256:{hash}"))?;
    package.metadata.package_hash = package.package_hash.clone();
    package.root_proof.package_claims = Some(serde_json::to_value(ResourcePackageClaims {
        resource_did: package.resource_did.clone(),
        resource_type: package.resource_type.clone(),
        version: package.package_version.clone(),
        did_document_hash: package.did_document_hash.clone(),
        metadata_hash: package.metadata_hash.clone(),
        package_hash: package.package_hash.clone(),
        hash_algorithm: package.hash_algorithm.clone(),
        lifecycle_state: package.metadata.lifecycle_state.clone(),
        authorized_domains: package.metadata.authorized_domains.clone(),
        bulletin_ref: None,
    })?);
    Ok(())
}

fn read_utf8_text_allowing_bom(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)?;
    Ok(text.strip_prefix('\u{feff}').unwrap_or(&text).to_owned())
}

async fn write_sync_history_store(state: &AppState, item: Value) -> Result<()> {
    if let Some(sqlite) = &state.sqlite {
        sqlite
            .upsert_json(
                "discovery.sync_history",
                &format!("{}", Utc::now().timestamp_nanos_opt().unwrap_or_default()),
                &item,
            )
            .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        postgres
            .upsert_json(
                "discovery.sync_history",
                &format!("{}", Utc::now().timestamp_nanos_opt().unwrap_or_default()),
                &item,
            )
            .await?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiscoveryQueryStatRecord {
    #[serde(rename = "statDate")]
    stat_date: String,
    #[serde(rename = "queryCount")]
    query_count: i64,
    #[serde(rename = "updatedAt")]
    updated_at: String,
}

async fn record_discovery_query_stat(state: &AppState) -> Result<()> {
    let today = Utc::now().date_naive().to_string();
    let updated_at = Utc::now();
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_QUERY_STATS_TABLE}(stat_date, query_count, updated_at)
            VALUES (?1, 1, ?2)
            ON CONFLICT(stat_date)
            DO UPDATE SET query_count = query_count + 1, updated_at = excluded.updated_at
            "#
        ))
        .bind(&today)
        .bind(updated_at.to_rfc3339())
        .execute(sqlite.pool())
        .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_QUERY_STATS_TABLE}(stat_date, query_count, updated_at)
            VALUES ($1, 1, $2)
            ON CONFLICT(stat_date)
            DO UPDATE SET query_count = {DISCOVERY_QUERY_STATS_TABLE}.query_count + 1, updated_at = excluded.updated_at
            "#
        ))
        .bind(&today)
        .bind(updated_at)
        .execute(postgres.pool())
        .await?;
        return Ok(());
    }
    let path = Path::new("query-stats").join(format!("{today}.json"));
    let mut record = state
        .data
        .read::<DiscoveryQueryStatRecord>(&path)
        .unwrap_or(DiscoveryQueryStatRecord {
            stat_date: today.clone(),
            query_count: 0,
            updated_at: String::new(),
        });
    record.query_count += 1;
    record.updated_at = updated_at.to_rfc3339();
    state.data.write(path, &record)?;
    Ok(())
}

async fn read_discovery_query_stats(state: &AppState) -> Result<Vec<DiscoveryQueryStatRecord>> {
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query_as::<_, (String, i64, String)>(&format!(
            "SELECT stat_date, query_count, updated_at::text FROM {DISCOVERY_QUERY_STATS_TABLE}"
        ))
        .fetch_all(sqlite.pool())
        .await?;
        return Ok(rows
            .into_iter()
            .map(
                |(stat_date, query_count, updated_at)| DiscoveryQueryStatRecord {
                    stat_date,
                    query_count,
                    updated_at,
                },
            )
            .collect());
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query_as::<_, (String, i64, String)>(&format!(
            "SELECT stat_date, query_count, updated_at::text FROM {DISCOVERY_QUERY_STATS_TABLE}"
        ))
        .fetch_all(postgres.pool())
        .await?;
        return Ok(rows
            .into_iter()
            .map(
                |(stat_date, query_count, updated_at)| DiscoveryQueryStatRecord {
                    stat_date,
                    query_count,
                    updated_at,
                },
            )
            .collect());
    }
    let mut records = Vec::new();
    let dir = state.data.root().join("query-stats");
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(records),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(relative) = path.strip_prefix(state.data.root()) else {
            continue;
        };
        if let Ok(record) = state.data.read(relative) {
            records.push(record);
        }
    }
    Ok(records)
}

async fn read_sync_cursor(state: &AppState) -> Result<i64> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(&format!(
            "SELECT COALESCE(state_value, sync_value) FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE state_key = ? OR sync_key = ?"
        ))
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .fetch_optional(sqlite.pool())
        .await?;
        return Ok(row
            .and_then(|row| row.get::<String, _>(0).parse::<i64>().ok())
            .unwrap_or(0));
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(&format!(
            "SELECT COALESCE(state_value::text, sync_value::text) FROM {DISCOVERY_SYNC_STATE_TABLE} WHERE state_key = $1 OR sync_key = $1"
        ))
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .fetch_optional(postgres.pool())
        .await?;
        return Ok(row
            .and_then(|row| row.get::<String, _>(0).parse::<i64>().ok())
            .unwrap_or(0));
    }
    Ok(state
        .index
        .read::<Value>("sync-state.json")
        .ok()
        .and_then(|value| value[DISCOVERY_CDN_CURSOR_KEY].as_i64())
        .unwrap_or(0))
}

async fn write_sync_cursor(state: &AppState, cursor: i64) -> Result<()> {
    let updated_at = Utc::now();
    let updated_at_text = updated_at.to_rfc3339();
    if let Some(sqlite) = &state.sqlite {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_SYNC_STATE_TABLE}(state_key, state_value, sync_key, sync_value, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                sync_key = excluded.sync_key,
                sync_value = excluded.sync_value,
                updated_at = excluded.updated_at
            "#
        ))
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .bind(cursor.to_string())
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .bind(cursor.to_string())
        .bind(&updated_at_text)
        .execute(sqlite.pool())
        .await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        sqlx::query(&format!(
            r#"
            INSERT INTO {DISCOVERY_SYNC_STATE_TABLE}(state_key, state_value, sync_key, sync_value, updated_at)
            VALUES ($1, $2::jsonb, $1, $2::jsonb, $3::timestamptz)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                sync_key = excluded.sync_key,
                sync_value = excluded.sync_value,
                updated_at = excluded.updated_at
            "#
        ))
        .bind(DISCOVERY_CDN_CURSOR_KEY)
        .bind(cursor.to_string())
        .bind(updated_at)
        .execute(postgres.pool())
        .await?;
        return Ok(());
    }
    state.index.write(
        "sync-state.json",
        &json!({ DISCOVERY_CDN_CURSOR_KEY: cursor }),
    )?;
    Ok(())
}

async fn write_rejected_packages(state: &AppState, rejected: &[Value]) -> Result<()> {
    if let Some(sqlite) = &state.sqlite {
        let now = Utc::now().to_rfc3339();
        let mut tx = sqlite.pool().begin().await?;
        for item in rejected {
            let reject_key = rejected_package_key(item);
            sqlx::query(&format!(
                r#"
                INSERT INTO {DISCOVERY_REJECTED_TABLE}(reject_key, item_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(reject_key)
                DO UPDATE SET item_json = excluded.item_json, updated_at = excluded.updated_at
                "#
            ))
            .bind(reject_key)
            .bind(serde_json::to_string(item)?)
            .bind(now.clone())
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        let now = Utc::now();
        let mut tx = postgres.pool().begin().await?;
        for item in rejected {
            let reject_key = rejected_package_key(item);
            sqlx::query(&format!(
                r#"
                INSERT INTO {DISCOVERY_REJECTED_TABLE}(reject_key, item_json, updated_at)
                VALUES ($1, $2::jsonb, $3::timestamptz)
                ON CONFLICT(reject_key)
                DO UPDATE SET item_json = excluded.item_json, updated_at = excluded.updated_at
                "#
            ))
            .bind(reject_key)
            .bind(serde_json::to_string(item)?)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    state.index.write("rejected-packages.json", &rejected)?;
    Ok(())
}

async fn merge_rejected_packages(state: &AppState, rejected: &[Value]) -> Result<()> {
    if rejected.is_empty() {
        return Ok(());
    }
    let keys = rejected
        .iter()
        .map(rejected_package_key)
        .collect::<Vec<_>>();
    let existing_by_key = read_rejected_packages_by_keys(state, &keys).await?;
    let merged = rejected
        .iter()
        .map(|item| {
            let key = rejected_package_key(item);
            merge_rejected_package_item(existing_by_key.get(&key), item)
        })
        .collect::<Vec<_>>();
    write_rejected_packages(state, &merged).await
}

async fn read_rejected_packages_by_keys(
    state: &AppState,
    reject_keys: &[String],
) -> Result<BTreeMap<String, Value>> {
    if reject_keys.is_empty() {
        return Ok(BTreeMap::new());
    }
    if let Some(sqlite) = &state.sqlite {
        let mut out = BTreeMap::new();
        for reject_key in reject_keys {
            if let Some(row) = sqlx::query(&format!(
                "SELECT item_json FROM {DISCOVERY_REJECTED_TABLE} WHERE reject_key = ?"
            ))
            .bind(reject_key)
            .fetch_optional(sqlite.pool())
            .await?
            {
                out.insert(
                    reject_key.clone(),
                    serde_json::from_str::<Value>(&row.get::<String, _>(0))?,
                );
            }
        }
        return Ok(out);
    }
    if let Some(postgres) = &state.postgres {
        let mut out = BTreeMap::new();
        for reject_key in reject_keys {
            if let Some(row) = sqlx::query(&format!(
                "SELECT item_json::text FROM {DISCOVERY_REJECTED_TABLE} WHERE reject_key = $1"
            ))
            .bind(reject_key)
            .fetch_optional(postgres.pool())
            .await?
            {
                out.insert(
                    reject_key.clone(),
                    serde_json::from_str::<Value>(&row.get::<String, _>(0))?,
                );
            }
        }
        return Ok(out);
    }
    let key_set = reject_keys.iter().collect::<HashSet<_>>();
    Ok(read_rejected_packages(state)
        .await?
        .into_iter()
        .filter_map(|item| {
            let key = rejected_package_key(&item);
            key_set.contains(&key).then_some((key, item))
        })
        .collect())
}

fn merge_rejected_package_item(existing: Option<&Value>, incoming: &Value) -> Value {
    let Some(existing) = existing else {
        return incoming.clone();
    };
    let mut merged = incoming.clone();
    if existing.get("reason").and_then(Value::as_str)
        == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        && incoming.get("reason").and_then(Value::as_str)
            == Some(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
    {
        merged["status"] = existing
            .get("status")
            .cloned()
            .unwrap_or_else(|| json!(REJECT_STATUS_LATE));
        merged["retryCount"] = existing
            .get("retryCount")
            .cloned()
            .unwrap_or_else(|| json!(0));
        merged["firstSeenAt"] = existing
            .get("firstSeenAt")
            .cloned()
            .unwrap_or_else(|| json!(Utc::now().to_rfc3339()));
        merged["lastRetryAt"] = existing.get("lastRetryAt").cloned().unwrap_or(Value::Null);
        merged["lastError"] = incoming
            .get("lastError")
            .cloned()
            .or_else(|| existing.get("lastError").cloned())
            .unwrap_or(Value::Null);
    }
    merged
}

fn rejected_package_key(item: &Value) -> String {
    format!(
        "{}:{}",
        rejected_package_resource_did(item),
        item["cursor"].as_i64().unwrap_or_default()
    )
}

fn rejected_package_resource_did(item: &Value) -> &str {
    item["resourceDid"]
        .as_str()
        .or_else(|| item["did"].as_str())
        .unwrap_or("unknown")
}

async fn delete_rejected_packages_by_did(state: &AppState, resource_dids: &[String]) -> Result<()> {
    if resource_dids.is_empty() {
        return Ok(());
    }
    if let Some(sqlite) = &state.sqlite {
        let mut tx = sqlite.pool().begin().await?;
        for resource_did in resource_dids {
            sqlx::query(&format!(
                "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE json_extract(item_json, '$.resourceDid') = ? OR json_extract(item_json, '$.did') = ?"
            ))
            .bind(resource_did)
            .bind(resource_did)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        let mut tx = postgres.pool().begin().await?;
        for resource_did in resource_dids {
            sqlx::query(&format!(
                "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE item_json->>'resourceDid' = $1 OR item_json->>'did' = $1"
            ))
            .bind(resource_did)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    let dids = resource_dids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let rejected = read_rejected_packages(state)
        .await?
        .into_iter()
        .filter(|item| !dids.contains(rejected_package_resource_did(item)))
        .collect::<Vec<_>>();
    state.index.write("rejected-packages.json", &rejected)?;
    Ok(())
}

async fn delete_stale_rejected_packages_for_indexed_did(
    state: &AppState,
    resource_did: &str,
    indexed_cursor: i64,
    indexed_version: Option<&str>,
) -> Result<()> {
    if resource_did.is_empty() || resource_did == "unknown" {
        return Ok(());
    }
    if let Some(sqlite) = &state.sqlite {
        let mut query = QueryBuilder::<Sqlite>::new(format!(
            "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE (json_extract(item_json, '$.resourceDid') = "
        ));
        query.push_bind(resource_did);
        query.push(" OR json_extract(item_json, '$.did') = ");
        query.push_bind(resource_did);
        query.push(") AND (json_extract(item_json, '$.cursor') IS NULL OR json_extract(item_json, '$.cursor') <= ");
        query.push_bind(indexed_cursor);
        if let Some(indexed_version) = indexed_version.filter(|version| !version.is_empty()) {
            query.push(" OR json_extract(item_json, '$.packageVersion') = ");
            query.push_bind(indexed_version);
        }
        query.push(")");
        query.build().execute(sqlite.pool()).await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        let mut query = QueryBuilder::<Postgres>::new(format!(
            "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE (item_json->>'resourceDid' = "
        ));
        query.push_bind(resource_did);
        query.push(" OR item_json->>'did' = ");
        query.push_bind(resource_did);
        query.push(") AND (item_json->>'cursor' IS NULL OR (item_json->>'cursor')::bigint <= ");
        query.push_bind(indexed_cursor);
        if let Some(indexed_version) = indexed_version.filter(|version| !version.is_empty()) {
            query.push(" OR item_json->>'packageVersion' = ");
            query.push_bind(indexed_version);
        }
        query.push(")");
        query.build().execute(postgres.pool()).await?;
        return Ok(());
    }
    let keys = read_rejected_packages(state)
        .await?
        .into_iter()
        .filter(|item| {
            is_stale_rejected_for_indexed_did(item, resource_did, indexed_cursor, indexed_version)
        })
        .map(|item| rejected_package_key(&item))
        .collect::<Vec<_>>();
    delete_rejected_packages(state, &keys).await
}

async fn delete_rejected_packages(state: &AppState, reject_keys: &[String]) -> Result<()> {
    if reject_keys.is_empty() {
        return Ok(());
    }
    if let Some(sqlite) = &state.sqlite {
        let mut tx = sqlite.pool().begin().await?;
        for reject_key in reject_keys {
            sqlx::query(&format!(
                "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE reject_key = ?"
            ))
            .bind(reject_key)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    if let Some(postgres) = &state.postgres {
        let mut tx = postgres.pool().begin().await?;
        for reject_key in reject_keys {
            sqlx::query(&format!(
                "DELETE FROM {DISCOVERY_REJECTED_TABLE} WHERE reject_key = $1"
            ))
            .bind(reject_key)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        return Ok(());
    }
    let reject_key_set = reject_keys.iter().collect::<HashSet<_>>();
    let rejected = read_rejected_packages(state)
        .await?
        .into_iter()
        .filter(|item| !reject_key_set.contains(&rejected_package_key(item)))
        .collect::<Vec<_>>();
    state.index.write("rejected-packages.json", &rejected)?;
    Ok(())
}

async fn read_latest_sync_history_item(state: &AppState) -> Result<Option<Value>> {
    if let Some(sqlite) = &state.sqlite {
        let row = sqlx::query(
            "SELECT value_json FROM json_records WHERE namespace = ? ORDER BY updated_at DESC, record_key DESC LIMIT 1",
        )
        .bind("discovery.sync_history")
        .fetch_optional(sqlite.pool())
        .await?;
        return row
            .map(|row| {
                serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(anyhow::Error::from)
            })
            .transpose();
    }
    if let Some(postgres) = &state.postgres {
        let row = sqlx::query(
            "SELECT value_json::text FROM json_records WHERE namespace = $1 ORDER BY updated_at DESC, record_key DESC LIMIT 1",
        )
        .bind("discovery.sync_history")
        .fetch_optional(postgres.pool())
        .await?;
        return row
            .map(|row| {
                serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(anyhow::Error::from)
            })
            .transpose();
    }
    Ok(state
        .index
        .read::<Vec<Value>>("sync-history.json")
        .unwrap_or_default()
        .into_iter()
        .last())
}

async fn read_sync_history_page(
    state: &AppState,
    after_key: Option<&str>,
    limit: u32,
) -> Result<KeyPage<Value>> {
    let fetch_limit = i64::from(limit.max(1) + 1);
    let mut rows = Vec::<(String, Value)>::new();
    if let Some(sqlite) = &state.sqlite {
        let db_rows = if let Some(after_key) = after_key {
            sqlx::query(
                "SELECT record_key, value_json FROM json_records WHERE namespace = ? AND record_key > ? ORDER BY record_key LIMIT ?",
            )
            .bind("discovery.sync_history")
            .bind(after_key)
            .bind(fetch_limit)
            .fetch_all(sqlite.pool())
            .await?
        } else {
            sqlx::query(
                "SELECT record_key, value_json FROM json_records WHERE namespace = ? ORDER BY record_key LIMIT ?",
            )
            .bind("discovery.sync_history")
            .bind(fetch_limit)
            .fetch_all(sqlite.pool())
            .await?
        };
        for row in db_rows {
            rows.push((
                row.get::<String, _>(0),
                serde_json::from_str::<Value>(&row.get::<String, _>(1))?,
            ));
        }
    } else if let Some(postgres) = &state.postgres {
        let db_rows = if let Some(after_key) = after_key {
            sqlx::query(
                "SELECT record_key, value_json::text FROM json_records WHERE namespace = $1 AND record_key > $2 ORDER BY record_key LIMIT $3",
            )
            .bind("discovery.sync_history")
            .bind(after_key)
            .bind(fetch_limit)
            .fetch_all(postgres.pool())
            .await?
        } else {
            sqlx::query(
                "SELECT record_key, value_json::text FROM json_records WHERE namespace = $1 ORDER BY record_key LIMIT $2",
            )
            .bind("discovery.sync_history")
            .bind(fetch_limit)
            .fetch_all(postgres.pool())
            .await?
        };
        for row in db_rows {
            rows.push((
                row.get::<String, _>(0),
                serde_json::from_str::<Value>(&row.get::<String, _>(1))?,
            ));
        }
    } else {
        rows = state
            .index
            .read::<Vec<Value>>("sync-history.json")
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, item)| ((index + 1).to_string(), item))
            .filter(|(key, _)| {
                after_key
                    .and_then(|cursor| cursor.parse::<usize>().ok())
                    .map(|cursor| key.parse::<usize>().unwrap_or_default() > cursor)
                    .unwrap_or(true)
            })
            .collect();
        rows.truncate(fetch_limit as usize);
    }
    key_page_from_rows(rows, limit)
}

async fn export_discovery_debug_snapshot(state: &AppState) -> Result<()> {
    if state.sqlite.is_none() && state.postgres.is_none() {
        return Ok(());
    }
    fs::create_dir_all(state.index.root())?;

    let resource_path = state.index.resolve("resource-capabilities.json");
    let resource_tmp_path = state.index.resolve("resource-capabilities.json.tmp");
    let mut resource_writer = BufWriter::new(fs::File::create(&resource_tmp_path)?);
    resource_writer.write_all(b"[\n")?;
    let mut first = true;
    let mut cursor = 0_i64;
    loop {
        let page =
            read_indexed_resource_page(state, cursor, None, DISCOVERY_DEFAULT_PAGE_SIZE).await?;
        if page.items.is_empty() {
            break;
        }
        for (_, package) in page.items {
            if !first {
                resource_writer.write_all(b",\n")?;
            }
            serde_json::to_writer_pretty(&mut resource_writer, &package)?;
            first = false;
        }
        if !page.has_more {
            break;
        }
        cursor = page.next_cursor;
    }
    resource_writer.write_all(b"\n]\n")?;
    resource_writer.flush()?;
    fs::rename(&resource_tmp_path, &resource_path)?;

    let rejected_path = state.index.resolve("rejected-packages.json");
    let rejected_tmp_path = state.index.resolve("rejected-packages.json.tmp");
    let mut rejected_writer = BufWriter::new(fs::File::create(&rejected_tmp_path)?);
    rejected_writer.write_all(b"[\n")?;
    let mut first = true;
    let mut key = None::<String>;
    loop {
        let page =
            read_rejected_packages_page(state, key.as_deref(), DISCOVERY_DEFAULT_PAGE_SIZE, None)
                .await?;
        if page.items.is_empty() {
            break;
        }
        for item in page.items {
            if !first {
                rejected_writer.write_all(b",\n")?;
            }
            serde_json::to_writer_pretty(&mut rejected_writer, &item)?;
            first = false;
        }
        if !page.has_more {
            break;
        }
        key = page.next_key;
    }
    rejected_writer.write_all(b"\n]\n")?;
    rejected_writer.flush()?;
    fs::rename(&rejected_tmp_path, &rejected_path)?;

    let history_path = state.index.resolve("sync-history.json");
    let history_tmp_path = state.index.resolve("sync-history.json.tmp");
    let mut history_writer = BufWriter::new(fs::File::create(&history_tmp_path)?);
    history_writer.write_all(b"[\n")?;
    let mut first = true;
    let mut key = None::<String>;
    loop {
        let page =
            read_sync_history_page(state, key.as_deref(), DISCOVERY_DEFAULT_PAGE_SIZE).await?;
        if page.items.is_empty() {
            break;
        }
        for item in page.items {
            if !first {
                history_writer.write_all(b",\n")?;
            }
            serde_json::to_writer_pretty(&mut history_writer, &item)?;
            first = false;
        }
        if !page.has_more {
            break;
        }
        key = page.next_key;
    }
    history_writer.write_all(b"\n]\n")?;
    history_writer.flush()?;
    fs::rename(&history_tmp_path, &history_path)?;
    Ok(())
}

async fn read_rejected_packages(state: &AppState) -> Result<Vec<Value>> {
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query(&format!(
            "SELECT item_json FROM {DISCOVERY_REJECTED_TABLE} ORDER BY updated_at, reject_key"
        ))
        .fetch_all(sqlite.pool())
        .await?;
        return rows
            .into_iter()
            .map(|row| {
                serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(anyhow::Error::from)
            })
            .collect();
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            "SELECT item_json::text FROM {DISCOVERY_REJECTED_TABLE} ORDER BY updated_at, reject_key"
        ))
        .fetch_all(postgres.pool())
        .await?;
        return rows
            .into_iter()
            .map(|row| {
                serde_json::from_str::<Value>(&row.get::<String, _>(0)).map_err(anyhow::Error::from)
            })
            .collect();
    }
    Ok(state
        .index
        .read("rejected-packages.json")
        .unwrap_or_default())
}

async fn read_rejected_packages_page(
    state: &AppState,
    after_key: Option<&str>,
    limit: u32,
    status: Option<&str>,
) -> Result<KeyPage<Value>> {
    let fetch_limit = i64::from(limit.max(1) + 1);
    let mut rows = Vec::<(String, Value)>::new();
    if let Some(sqlite) = &state.sqlite {
        let mut builder = QueryBuilder::<Sqlite>::new(format!(
            "SELECT reject_key, item_json FROM {DISCOVERY_REJECTED_TABLE}"
        ));
        let mut separated = false;
        if let Some(after_key) = after_key {
            builder.push(" WHERE reject_key > ");
            builder.push_bind(after_key);
            separated = true;
        }
        if let Some(status) = status {
            builder.push(if separated { " AND " } else { " WHERE " });
            builder.push("COALESCE(json_extract(item_json, '$.status'), ");
            builder.push_bind(REJECT_STATUS_LATE);
            builder.push(") = ");
            builder.push_bind(status);
        }
        builder.push(" ORDER BY reject_key LIMIT ");
        builder.push_bind(fetch_limit);
        let db_rows = builder.build().fetch_all(sqlite.pool()).await?;
        for row in db_rows {
            let item = serde_json::from_str::<Value>(&row.get::<String, _>(1))?;
            rows.push((row.get::<String, _>(0), item));
        }
    } else if let Some(postgres) = &state.postgres {
        let mut builder = QueryBuilder::<Postgres>::new(format!(
            "SELECT reject_key, item_json::text FROM {DISCOVERY_REJECTED_TABLE}"
        ));
        let mut separated = false;
        if let Some(after_key) = after_key {
            builder.push(" WHERE reject_key > ");
            builder.push_bind(after_key);
            separated = true;
        }
        if let Some(status) = status {
            builder.push(if separated { " AND " } else { " WHERE " });
            builder.push("COALESCE(item_json->>'status', ");
            builder.push_bind(REJECT_STATUS_LATE);
            builder.push(") = ");
            builder.push_bind(status);
        }
        builder.push(" ORDER BY reject_key LIMIT ");
        builder.push_bind(fetch_limit);
        let db_rows = builder.build().fetch_all(postgres.pool()).await?;
        for row in db_rows {
            let item = serde_json::from_str::<Value>(&row.get::<String, _>(1))?;
            rows.push((row.get::<String, _>(0), item));
        }
    } else {
        rows = state
            .index
            .read::<Vec<Value>>("rejected-packages.json")
            .unwrap_or_default()
            .into_iter()
            .map(|item| (rejected_package_key(&item), item))
            .filter(|(key, item)| {
                after_key
                    .map(|cursor| key.as_str() > cursor)
                    .unwrap_or(true)
                    && status_matches_rejected_item(item, status)
            })
            .collect();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        rows.truncate(fetch_limit as usize);
    }
    key_page_from_rows(rows, limit)
}

fn key_page_from_rows<T>(mut rows: Vec<(String, T)>, limit: u32) -> Result<KeyPage<T>> {
    let page_limit = limit.max(1) as usize;
    let has_more = rows.len() > page_limit;
    if has_more {
        rows.truncate(page_limit);
    }
    let next_key = rows.last().map(|(key, _)| key.clone());
    Ok(KeyPage {
        items: rows.into_iter().map(|(_, item)| item).collect(),
        next_key,
        has_more,
    })
}

fn status_matches_rejected_item(item: &Value, status: Option<&str>) -> bool {
    status
        .map(|status| rejected_package_status(item) == status)
        .unwrap_or(true)
}

async fn rejected_package_retry_summary_from_store(state: &AppState) -> Result<Value> {
    if let Some(sqlite) = &state.sqlite {
        let rows = sqlx::query(&format!(
            r#"
            SELECT COALESCE(json_extract(item_json, '$.status'), ?), COUNT(*)
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE json_extract(item_json, '$.reason') = ?
            GROUP BY COALESCE(json_extract(item_json, '$.status'), ?)
            "#
        ))
        .bind(REJECT_STATUS_LATE)
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .bind(REJECT_STATUS_LATE)
        .fetch_all(sqlite.pool())
        .await?;
        let late_count = rows
            .iter()
            .find(|row| row.get::<Option<String>, _>(0).as_deref() == Some(REJECT_STATUS_LATE))
            .map(|row| row.get::<i64, _>(1) as usize)
            .unwrap_or_default();
        let dead_count = rows
            .iter()
            .find(|row| row.get::<Option<String>, _>(0).as_deref() == Some(REJECT_STATUS_DEAD))
            .map(|row| row.get::<i64, _>(1) as usize)
            .unwrap_or_default();
        return Ok(rejected_package_retry_summary_from_counts(
            late_count, dead_count,
        ));
    }
    if let Some(postgres) = &state.postgres {
        let rows = sqlx::query(&format!(
            r#"
            SELECT COALESCE(item_json->>'status', $1), COUNT(*)
            FROM {DISCOVERY_REJECTED_TABLE}
            WHERE item_json->>'reason' = $2
            GROUP BY COALESCE(item_json->>'status', $1)
            "#
        ))
        .bind(REJECT_STATUS_LATE)
        .bind(REJECT_REASON_RESOURCE_PACKAGE_UNAVAILABLE)
        .fetch_all(postgres.pool())
        .await?;
        let late_count = rows
            .iter()
            .find(|row| row.get::<Option<String>, _>(0).as_deref() == Some(REJECT_STATUS_LATE))
            .map(|row| row.get::<i64, _>(1) as usize)
            .unwrap_or_default();
        let dead_count = rows
            .iter()
            .find(|row| row.get::<Option<String>, _>(0).as_deref() == Some(REJECT_STATUS_DEAD))
            .map(|row| row.get::<i64, _>(1) as usize)
            .unwrap_or_default();
        return Ok(rejected_package_retry_summary_from_counts(
            late_count, dead_count,
        ));
    }
    Ok(rejected_package_retry_summary(
        &state
            .index
            .read::<Vec<Value>>("rejected-packages.json")
            .unwrap_or_default(),
    ))
}

fn validate_resource_package_for_index(
    package: &ResourcePackage,
) -> std::result::Result<(), String> {
    package
        .did_document
        .validate_oan_resource()
        .map_err(|err| err.to_string())?;
    if verify_did_document_proof_standard_blocking(&package.did_document).is_err() {
        verify_did_document_proof(&package.did_document)
            .map_err(|_| "did_document_proof_invalid".to_owned())?;
    }
    package
        .verify_did_document_hash()
        .and_then(|_| package.verify_metadata_hash())
        .and_then(|_| package.verify_package_hash())
        .and_then(|_| package.verify_resource_type_consistency())
        .and_then(|_| package.verify_metadata_consistency())
        .and_then(|_| package.verify_root_claim_binding())
        .map_err(|err| err.to_string())?;
    if package.metadata.lifecycle_state != "active" {
        return Err("resource_not_active".to_owned());
    }
    if package.metadata.authorized_domains.is_empty() {
        return Err("resource_domains_required".to_owned());
    }
    validate_authorized_domain_list(&package.metadata.authorized_domains)?;
    Ok(())
}

fn resource_matches_explicit_constraints(
    package: &ResourcePackage,
    query: &ResourceDiscoveryQuery,
) -> bool {
    if query
        .resource_type
        .as_ref()
        .is_some_and(|resource_type| resource_type != &package.resource_type)
    {
        return false;
    }
    if query
        .version
        .as_ref()
        .is_some_and(|version| version != &package.package_version)
    {
        return false;
    }
    if !query.capability_tags.is_empty()
        && !query
            .capability_tags
            .iter()
            .all(|tag| package.metadata.capability_tags.contains(tag))
    {
        return false;
    }
    if query.protocol.as_ref().is_some_and(|protocol| {
        let expected = protocol.to_ascii_lowercase();
        !package.metadata.protocol_bindings.iter().any(|binding| {
            binding["protocol"]
                .as_str()
                .map(|value| value.eq_ignore_ascii_case(&expected))
                .unwrap_or(false)
        }) && !package.metadata.services.iter().any(|service| {
            service
                .protocol
                .as_ref()
                .map(|value| value.eq_ignore_ascii_case(&expected))
                .unwrap_or(false)
        })
    }) {
        return false;
    }
    true
}

fn resource_matches_exact_did_query(
    package: &ResourcePackage,
    query: &ResourceDiscoveryQuery,
) -> bool {
    let mut constraint_query = query.clone();
    constraint_query.query = None;
    resource_matches_explicit_constraints(package, &constraint_query)
}

fn discovery_authorized_domains(bulletin: &Value, discovery_did: &str) -> Vec<String> {
    bulletin["events"]
        .as_array()
        .and_then(|events| {
            events
                .iter()
                .rev()
                .find(|event| {
                    event["subjectDid"] == discovery_did
                        && event["eventType"] == "DISCOVERY_NODE_DOMAINS_UPDATED"
                })
                .map(|event| event["payload"].clone())
        })
        .and_then(|payload| payload["authorizedDomains"].as_array().cloned())
        .map(|values| {
            values
                .into_iter()
                .filter_map(|value| value.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

