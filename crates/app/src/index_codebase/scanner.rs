//! File discovery and ignore policy for `index_codebase`.

use super::types::{IndexCodebaseDeps, IndexCodebaseInput};
use semantic_code_ports::{FileSystemEntryKind, FileSystemPortExt, IgnoreMatchInput};
use semantic_code_shared::{RequestContext, Result};
use std::collections::{HashSet, VecDeque};

// TODO: refactor repeated optional logger/telemetry checks with a helper mapper.
const CONTEXT_IGNORE_FILE: &str = ".contextignore";

pub async fn load_index_files(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    input: &IndexCodebaseInput,
) -> Result<Vec<Box<str>>> {
    let ignore_patterns = load_ignore_patterns(ctx, deps, input).await?;
    let raw_files = if let Some(file_list) = input.file_list.as_ref() {
        let mut files: Vec<Box<str>> = file_list
            .iter()
            .map(|path| normalize_relative_path(path).into_boxed_str())
            .collect();
        files.sort();
        files
    } else {
        scan_code_files(ctx, deps, input, &ignore_patterns).await?
    };

    filter_files(ctx, deps, raw_files, input, &ignore_patterns)
}

pub async fn scan_code_files(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    input: &IndexCodebaseInput,
    ignore_patterns: &[Box<str>],
) -> Result<Vec<Box<str>>> {
    let supported_extensions = normalize_extensions(input.supported_extensions.as_ref());
    let filter_by_ext = !supported_extensions.is_empty();
    let fs = deps.filesystem.session(input.codebase_root.clone());

    let mut dirs: VecDeque<String> = VecDeque::from([String::from(".")]);
    let mut files: Vec<String> = Vec::new();

    while let Some(dir) = dirs.pop_front() {
        ctx.ensure_not_cancelled("index_codebase.scan")?;

        let safe_dir = deps.path_policy.to_safe_relative_path(&dir)?;
        let entries = match fs.read_dir(ctx, safe_dir).await {
            Ok(entries) => entries,
            Err(error) => {
                if error.is_cancelled() {
                    return Err(error);
                }
                if let Some(logger) = deps.logger.as_ref() {
                    logger.warn(
                        "index.scan.dir_read_failed",
                        "Cannot read directory during scan",
                        None,
                    );
                }
                continue;
            },
        };

        let mut sorted = entries;
        sorted.sort_by(|a, b| a.name.cmp(&b.name));

        for entry in sorted {
            ctx.ensure_not_cancelled("index_codebase.scan_entry")?;

            let rel = join_relative(&dir, entry.name.as_ref());
            if deps.ignore.is_ignored(&IgnoreMatchInput {
                ignore_patterns: ignore_patterns.to_vec(),
                relative_path: rel.clone().into_boxed_str(),
            }) {
                continue;
            }

            match entry.kind {
                FileSystemEntryKind::Directory => {
                    dirs.push_back(rel);
                },
                FileSystemEntryKind::File => {
                    let ext = file_extension_of(&rel);
                    if filter_by_ext {
                        let Some(ext) = ext.as_deref() else {
                            continue;
                        };
                        if !supported_extensions.contains(ext) {
                            continue;
                        }
                    }
                    files.push(rel);
                    if input.max_files.is_some_and(|max| files.len() >= max.get()) {
                        let mut out = files
                            .into_iter()
                            .map(String::into_boxed_str)
                            .collect::<Vec<_>>();
                        out.sort();
                        return Ok(out);
                    }
                },
                FileSystemEntryKind::Other => {},
            }
        }
    }

    let mut out = files
        .into_iter()
        .map(String::into_boxed_str)
        .collect::<Vec<_>>();
    out.sort();
    Ok(out)
}

pub fn filter_files(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    raw_files: Vec<Box<str>>,
    input: &IndexCodebaseInput,
    ignore_patterns: &[Box<str>],
) -> Result<Vec<Box<str>>> {
    ctx.ensure_not_cancelled("index_codebase.filter_files")?;

    let supported_extensions = normalize_extensions(input.supported_extensions.as_ref());
    let filter_by_ext = !supported_extensions.is_empty();

    let mut files = Vec::new();
    for rel in raw_files {
        ctx.ensure_not_cancelled("index_codebase.filter_files")?;

        let normalized = normalize_relative_path(rel.as_ref());
        if deps.ignore.is_ignored(&IgnoreMatchInput {
            ignore_patterns: ignore_patterns.to_vec(),
            relative_path: normalized.clone().into_boxed_str(),
        }) {
            continue;
        }

        let ext = file_extension_of(&normalized);
        if filter_by_ext {
            let Some(ext) = ext.as_deref() else {
                continue;
            };
            if !supported_extensions.contains(ext) {
                continue;
            }
        }

        files.push(normalized.into_boxed_str());
        if input.max_files.is_some_and(|max| files.len() >= max.get()) {
            break;
        }
    }

    files.sort();
    Ok(files)
}

pub async fn load_ignore_patterns(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    input: &IndexCodebaseInput,
) -> Result<Vec<Box<str>>> {
    let mut patterns = input.ignore_patterns.clone().unwrap_or_default();
    patterns.push(CONTEXT_IGNORE_FILE.into());

    let safe_path = deps
        .path_policy
        .to_safe_relative_path(CONTEXT_IGNORE_FILE)?;
    let fs = deps.filesystem.session(input.codebase_root.clone());
    match fs.read_file_text(ctx, safe_path).await {
        Ok(contents) => {
            patterns.extend(parse_context_ignore(&contents));
        },
        Err(error) => {
            if error.is_cancelled() {
                return Err(error);
            }
            if error.code != semantic_code_shared::ErrorCode::not_found()
                && let Some(logger) = deps.logger.as_ref()
            {
                logger.warn(
                    "index.ignore.read_failed",
                    "Failed to read .contextignore; continuing without it",
                    None,
                );
            }
        },
    }

    patterns.sort();
    patterns.dedup();
    Ok(patterns)
}

fn parse_context_ignore(contents: &str) -> Vec<Box<str>> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.to_owned().into_boxed_str())
        .collect()
}

fn normalize_extensions(values: Option<&Vec<Box<str>>>) -> HashSet<Box<str>> {
    let mut out = HashSet::new();
    let Some(values) = values else {
        return out;
    };
    for raw in values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let trimmed = trimmed.trim_start_matches('.');
        if trimmed.is_empty() {
            continue;
        }
        out.insert(trimmed.to_ascii_lowercase().into_boxed_str());
    }
    out
}

pub fn normalize_relative_path(path: &str) -> String {
    let mut out = String::new();
    let mut prev_slash = false;
    for ch in path.chars() {
        let ch = if ch == '\\' { '/' } else { ch };
        if ch == '/' {
            if prev_slash {
                continue;
            }
            prev_slash = true;
            out.push('/');
        } else {
            prev_slash = false;
            out.push(ch);
        }
    }
    if out.is_empty() { ".".to_string() } else { out }
}

pub fn file_extension_of(path: &str) -> Option<Box<str>> {
    let file = path.rsplit('/').next().unwrap_or(path);
    let (_, ext) = file.rsplit_once('.')?;
    if ext.is_empty() {
        return None;
    }
    Some(ext.to_ascii_lowercase().into_boxed_str())
}

fn join_relative(parent: &str, child: &str) -> String {
    if parent == "." || parent.trim().is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}
