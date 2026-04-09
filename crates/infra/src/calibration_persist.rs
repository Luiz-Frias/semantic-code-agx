//! SQLite WAL-backed EMA persistence for calibrated BQ1 observation updates.

#[cfg(test)]
use crate::cli_calibration::write_calibration;
use crate::cli_calibration::{calibration_path, read_calibration};
use crate::{InfraError, InfraResult};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use semantic_code_domain::{CalibrationState, EmaState, SearchStats};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, UNIX_EPOCH};
use std::{fs, thread};

const DRIFT_TOLERANCE: f32 = 0.15;
const HAMMING_SKIPPED_KEY: &str = "hammingSkipped";
const CALIBRATION_SQLITE_FILE_NAME: &str = "calibration_ema.db";
const CALIBRATION_SQLITE_WAL_SUFFIX: &str = "-wal";
const CALIBRATION_SQLITE_SHM_SUFFIX: &str = "-shm";
#[cfg(not(test))]
const LOCK_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(test)]
const LOCK_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
const OBSERVATION_RETRY_BACKOFF_MS: [u64; 5] = [2, 4, 8, 16, 32];
static CALIBRATION_CACHE: OnceLock<Mutex<HashMap<PathBuf, CalibrationCacheEntry>>> =
    OnceLock::new();

/// Observation output returned to the caller for logging/drift checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmaObservationResult {
    /// Calibrated skip-rate baseline.
    pub calibrated_skip_rate: f32,
    /// Observed skip-rate for the current query.
    pub observed_skip_rate: f32,
    /// Effective EMA after the observation.
    pub ema_skip_rate: f32,
    /// Effective observed sample count.
    pub samples_seen: u64,
    /// Whether drift was detected using the calibrated baseline.
    pub drift_detected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SqliteEmaRow {
    calibrated_at_ms: u64,
    samples_seen: u64,
    alpha_bits: u32,
    current_skip_rate_bits: u32,
}

#[derive(Debug, Clone)]
struct CalibrationCacheEntry {
    file_revision: Option<u128>,
    calibration: Option<CalibrationState>,
}

impl SqliteEmaRow {
    const fn into_ema(self, expected_calibrated_at_ms: u64) -> Option<EmaState> {
        if self.calibrated_at_ms != expected_calibrated_at_ms {
            return None;
        }
        Some(EmaState {
            alpha: f32::from_bits(self.alpha_bits),
            current_skip_rate: f32::from_bits(self.current_skip_rate_bits),
            samples_seen: self.samples_seen,
        })
    }
}

/// Observe and persist EMA state using `SQLite` WAL-backed transactions.
pub fn apply_ema_observation(
    codebase_root: &Path,
    search_stats: &SearchStats,
) -> InfraResult<Option<EmaObservationResult>> {
    let Some(skip_rate) = extract_skip_rate(search_stats) else {
        return Ok(None);
    };

    for (attempt, backoff_ms) in OBSERVATION_RETRY_BACKOFF_MS.iter().enumerate() {
        match observe_with_sqlite_wal(codebase_root, skip_rate) {
            Ok(observation) => return Ok(observation),
            Err(error)
                if attempt + 1 < OBSERVATION_RETRY_BACKOFF_MS.len()
                    && is_retryable_sqlite_lock_error(&error) =>
            {
                thread::sleep(Duration::from_millis(*backoff_ms));
            },
            Err(error) => return Err(error),
        }
    }

    Err(ErrorEnvelope::unexpected(
        ErrorCode::internal(),
        "calibration observation retry loop exhausted unexpectedly",
        ErrorClass::NonRetriable,
    ))
}

/// Remove EMA sidecars created by `SQLite` persistence strategy.
pub fn clear_ema_sidecars(root: &Path) -> InfraResult<()> {
    let sqlite_path = sqlite_db_path(root);
    remove_file_if_exists(&sqlite_path)?;
    remove_file_if_exists(&PathBuf::from(format!(
        "{}{}",
        sqlite_path.display(),
        CALIBRATION_SQLITE_WAL_SUFFIX
    )))?;
    remove_file_if_exists(&PathBuf::from(format!(
        "{}{}",
        sqlite_path.display(),
        CALIBRATION_SQLITE_SHM_SUFFIX
    )))?;
    Ok(())
}

fn observe_with_sqlite_wal(
    root: &Path,
    skip_rate: f32,
) -> InfraResult<Option<EmaObservationResult>> {
    let Some(mut calibration) = read_calibration_cached(root)? else {
        return Ok(None);
    };

    let mut connection = open_sqlite_connection(&sqlite_db_path(root))?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("calibration sqlite transaction start failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    let stored_row = read_sqlite_ema_row(&transaction)?;

    let mut ema = stored_row
        .and_then(|row| row.into_ema(calibration.calibrated_at_ms))
        .or_else(|| calibration.ema.clone())
        .unwrap_or_default();
    ema.observe(skip_rate);
    write_sqlite_ema_row(&transaction, calibration.calibrated_at_ms, &ema)?;
    transaction.commit().map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::io(),
            format!("calibration sqlite commit failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;

    calibration.ema = Some(ema);
    Ok(Some(observation_from_state(skip_rate, &calibration)))
}

fn open_sqlite_connection(path: &Path) -> InfraResult<Connection> {
    if let Some(context_dir) = path.parent() {
        fs::create_dir_all(context_dir)?;
    }

    let connection = Connection::open(path).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::io(),
            format!("calibration sqlite open failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    connection
        .busy_timeout(LOCK_WAIT_TIMEOUT)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::timeout(),
                format!("calibration sqlite busy-timeout failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    connection
        .pragma_update(None, "journal_mode", "WAL")
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("calibration sqlite WAL setup failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS ema_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                calibrated_at_ms INTEGER NOT NULL,
                alpha REAL NOT NULL,
                current_skip_rate REAL NOT NULL,
                samples_seen INTEGER NOT NULL
            );",
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("calibration sqlite schema setup failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    Ok(connection)
}

fn read_sqlite_ema_row(
    transaction: &rusqlite::Transaction<'_>,
) -> InfraResult<Option<SqliteEmaRow>> {
    transaction
        .query_row(
            "SELECT calibrated_at_ms, alpha, current_skip_rate, samples_seen
             FROM ema_state
             WHERE id = 1",
            [],
            |row| {
                let calibrated_at_ms = row.get::<_, i64>(0)?.cast_unsigned();
                let alpha = row.get::<_, f32>(1)?;
                let current_skip_rate = row.get::<_, f32>(2)?;
                let samples_seen = row.get::<_, i64>(3)?.cast_unsigned();
                Ok(SqliteEmaRow {
                    calibrated_at_ms,
                    samples_seen,
                    alpha_bits: alpha.to_bits(),
                    current_skip_rate_bits: current_skip_rate.to_bits(),
                })
            },
        )
        .optional()
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("calibration sqlite read failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })
}

fn write_sqlite_ema_row(
    transaction: &rusqlite::Transaction<'_>,
    calibrated_at_ms: u64,
    ema: &EmaState,
) -> InfraResult<()> {
    transaction
        .execute(
            "INSERT INTO ema_state (
                id,
                calibrated_at_ms,
                alpha,
                current_skip_rate,
                samples_seen
            ) VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                calibrated_at_ms = excluded.calibrated_at_ms,
                alpha = excluded.alpha,
                current_skip_rate = excluded.current_skip_rate,
                samples_seen = excluded.samples_seen",
            params![
                calibrated_at_ms.cast_signed(),
                f64::from(ema.alpha),
                f64::from(ema.current_skip_rate),
                ema.samples_seen.cast_signed(),
            ],
        )
        .map(|_| ())
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("calibration sqlite write failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "skip-rate values are bounded to [0, 1]"
)]
fn extract_skip_rate(search_stats: &SearchStats) -> Option<f32> {
    let hamming_skipped = search_stats.extra.get(HAMMING_SKIPPED_KEY).copied()?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "expansion counts are expected to fit in precise f64 range"
    )]
    let expansions = search_stats.expansions.unwrap_or(0) as f64;
    let total = hamming_skipped + expansions;
    if total <= 0.0 {
        return None;
    }
    Some((hamming_skipped / total) as f32)
}

fn observation_from_state(skip_rate: f32, calibration: &CalibrationState) -> EmaObservationResult {
    let (ema_skip_rate, samples_seen) = calibration
        .ema
        .as_ref()
        .map_or((0.0, 0), |ema| (ema.current_skip_rate, ema.samples_seen));
    EmaObservationResult {
        calibrated_skip_rate: calibration.skip_rate,
        observed_skip_rate: skip_rate,
        ema_skip_rate,
        samples_seen,
        drift_detected: calibration.is_drifted(DRIFT_TOLERANCE),
    }
}

fn remove_file_if_exists(path: &Path) -> InfraResult<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(InfraError::from(error)),
    }
}

fn is_retryable_sqlite_lock_error(error: &InfraError) -> bool {
    error.code == ErrorCode::timeout()
        || (error.code == ErrorCode::io()
            && error
                .message
                .to_ascii_lowercase()
                .contains("database is locked"))
}

fn sqlite_db_path(root: &Path) -> PathBuf {
    context_dir(root).join(CALIBRATION_SQLITE_FILE_NAME)
}

fn context_dir(root: &Path) -> PathBuf {
    calibration_path(root)
        .parent()
        .map_or_else(|| root.to_path_buf(), Path::to_path_buf)
}

fn read_calibration_cached(root: &Path) -> InfraResult<Option<CalibrationState>> {
    let path = calibration_path(root);
    let file_revision = calibration_file_revision(&path)?;
    let cache = calibration_cache();
    {
        let guard = lock_calibration_cache(cache);
        if let Some(entry) = guard.get(&path)
            && entry.file_revision == file_revision
        {
            return Ok(entry.calibration.clone());
        }
    }

    let calibration = read_calibration(root)?;
    {
        let mut guard = lock_calibration_cache(cache);
        guard.insert(
            path,
            CalibrationCacheEntry {
                file_revision,
                calibration: calibration.clone(),
            },
        );
    }
    Ok(calibration)
}

fn calibration_cache() -> &'static Mutex<HashMap<PathBuf, CalibrationCacheEntry>> {
    CALIBRATION_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_calibration_cache(
    cache: &'static Mutex<HashMap<PathBuf, CalibrationCacheEntry>>,
) -> MutexGuard<'static, HashMap<PathBuf, CalibrationCacheEntry>> {
    match cache.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("calibration cache lock poisoned; continuing with inner state");
            poisoned.into_inner()
        },
    }
}

fn calibration_file_revision(path: &Path) -> InfraResult<Option<u128>> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(InfraError::from(error)),
    };
    let modified = metadata.modified().map_err(InfraError::from)?;
    Ok(Some(
        modified
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::{env, thread};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    fn sample_calibration() -> CalibrationState {
        CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 1_700_000_000_000,
            dimension: 1024,
            corpus_size: 50_000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        }
    }

    fn stats_with_skip_rate(hamming_skipped: f64, expansions: u64) -> SearchStats {
        let mut extra = std::collections::BTreeMap::new();
        extra.insert(HAMMING_SKIPPED_KEY.into(), hamming_skipped);
        SearchStats {
            expansions: Some(expansions),
            kernel: "dfrr".into(),
            extra,
            kernel_search_duration_ns: None,
            index_size: None,
        }
    }

    fn run_concurrent_updates(workers: usize, updates_per_worker: usize) -> InfraResult<u64> {
        let root = temp_dir("ema-persist");
        fs::create_dir_all(&root)?;
        write_calibration(&root, &sample_calibration())?;
        clear_ema_sidecars(&root)?;

        let shared_root = Arc::new(root);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let root = Arc::clone(&shared_root);
            handles.push(thread::spawn(move || -> InfraResult<()> {
                let stats = stats_with_skip_rate(30.0, 70);
                for _ in 0..updates_per_worker {
                    let _ = apply_ema_observation(&root, &stats)?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "EMA worker thread panicked",
                    ErrorClass::NonRetriable,
                )
            })??;
        }

        read_samples_seen(&shared_root)
    }

    fn read_samples_seen(root: &Path) -> InfraResult<u64> {
        let mut state = read_calibration(root)?.ok_or_else(|| {
            ErrorEnvelope::expected(ErrorCode::not_found(), "missing calibration")
        })?;
        let connection = Connection::open(sqlite_db_path(root)).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::io(),
                format!("sqlite open failed: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
        let row: Option<(u64, f32, f32, u64)> = connection
            .query_row(
                "SELECT calibrated_at_ms, alpha, current_skip_rate, samples_seen
                 FROM ema_state
                 WHERE id = 1",
                [],
                |row| {
                    let calibrated_at_ms = row.get::<_, i64>(0)?.cast_unsigned();
                    let alpha = row.get::<_, f32>(1)?;
                    let current_skip_rate = row.get::<_, f32>(2)?;
                    let samples_seen = row.get::<_, i64>(3)?.cast_unsigned();
                    Ok((calibrated_at_ms, alpha, current_skip_rate, samples_seen))
                },
            )
            .optional()
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::io(),
                    format!("sqlite read failed: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;

        if let Some((calibrated_at_ms, alpha, current_skip_rate, samples_seen)) = row
            && calibrated_at_ms == state.calibrated_at_ms
        {
            state.ema = Some(EmaState {
                alpha,
                current_skip_rate,
                samples_seen,
            });
        }

        Ok(state.ema.as_ref().map_or(0, |ema| ema.samples_seen))
    }

    #[test]
    fn sqlite_wal_preserves_all_updates_under_concurrency() -> InfraResult<()> {
        let workers = 8usize;
        let updates_per_worker = 125usize;
        let samples_seen = run_concurrent_updates(workers, updates_per_worker)?;
        assert_eq!(
            samples_seen,
            u64::try_from(workers * updates_per_worker).unwrap_or(0)
        );
        Ok(())
    }

    #[test]
    fn clear_sidecars_removes_sqlite_files() -> InfraResult<()> {
        let root = temp_dir("ema-sidecars");
        fs::create_dir_all(&root)?;
        write_calibration(&root, &sample_calibration())?;

        fs::write(sqlite_db_path(&root), b"sqlite")?;
        fs::write(
            PathBuf::from(format!(
                "{}{}",
                sqlite_db_path(&root).display(),
                CALIBRATION_SQLITE_WAL_SUFFIX
            )),
            b"sqlite-wal",
        )?;
        fs::write(
            PathBuf::from(format!(
                "{}{}",
                sqlite_db_path(&root).display(),
                CALIBRATION_SQLITE_SHM_SUFFIX
            )),
            b"sqlite-shm",
        )?;

        clear_ema_sidecars(&root)?;
        assert!(!sqlite_db_path(&root).exists());
        assert!(
            !PathBuf::from(format!(
                "{}{}",
                sqlite_db_path(&root).display(),
                CALIBRATION_SQLITE_WAL_SUFFIX
            ))
            .exists()
        );
        assert!(
            !PathBuf::from(format!(
                "{}{}",
                sqlite_db_path(&root).display(),
                CALIBRATION_SQLITE_SHM_SUFFIX
            ))
            .exists()
        );
        Ok(())
    }

    #[test]
    fn cached_calibration_refreshes_after_file_change() -> InfraResult<()> {
        let root = temp_dir("ema-cache-refresh");
        fs::create_dir_all(&root)?;

        let initial = sample_calibration();
        write_calibration(&root, &initial)?;
        let loaded_initial = read_calibration_cached(&root)?.expect("calibration should exist");
        assert_eq!(loaded_initial.threshold, initial.threshold);

        thread::sleep(Duration::from_millis(2));
        let mut updated = sample_calibration();
        updated.threshold = 0.77;
        write_calibration(&root, &updated)?;

        let loaded_updated = read_calibration_cached(&root)?.expect("calibration should exist");
        assert_eq!(loaded_updated.threshold, updated.threshold);
        Ok(())
    }

    #[test]
    fn cached_calibration_transitions_from_missing_to_present() -> InfraResult<()> {
        let root = temp_dir("ema-cache-missing");
        fs::create_dir_all(&root)?;

        assert!(read_calibration_cached(&root)?.is_none());
        thread::sleep(Duration::from_millis(2));
        let state = sample_calibration();
        write_calibration(&root, &state)?;

        let loaded = read_calibration_cached(&root)?.expect("calibration should exist");
        assert_eq!(loaded.calibrated_at_ms, state.calibrated_at_ms);
        Ok(())
    }
}
