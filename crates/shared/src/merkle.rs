//! Merkle DAG utilities for snapshotting.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// A Merkle DAG node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleDagNode {
    /// Stable node identifier (hash of `data`).
    pub id: String,
    /// Hash value (same as `id`).
    pub hash: String,
    /// Raw node payload.
    pub data: String,
    /// Parent node ids.
    pub parents: Vec<String>,
    /// Child node ids.
    pub children: Vec<String>,
}

/// Serialized representation of a Merkle DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleDagSerialized {
    /// Node entries as `(id, node)` tuples.
    pub nodes: Vec<(String, MerkleDagNode)>,
    /// Root node identifiers.
    #[serde(rename = "rootIds")]
    pub root_ids: Vec<String>,
}

/// Merkle DAG diff result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MerkleDagDiff {
    /// Added node ids.
    pub added: Vec<String>,
    /// Removed node ids.
    pub removed: Vec<String>,
    /// Modified node ids (same id, different data).
    pub modified: Vec<String>,
}

impl MerkleDagDiff {
    /// Return true when no changes are present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
}

/// Merkle DAG implementation.
#[derive(Debug, Clone, Default)]
pub struct MerkleDag {
    nodes: BTreeMap<String, MerkleDagNode>,
    root_ids: Vec<String>,
}

impl MerkleDag {
    /// Create a new empty DAG.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            root_ids: Vec::new(),
        }
    }

    /// Add a node, optionally attached to a parent.
    pub fn add_node(&mut self, data: &str, parent_id: Option<&str>) -> String {
        let node_id = hash_data(data);
        let mut node = MerkleDagNode {
            id: node_id.clone(),
            hash: node_id.clone(),
            data: data.to_owned(),
            parents: Vec::new(),
            children: Vec::new(),
        };

        if let Some(parent_id) = parent_id {
            if let Some(parent) = self.nodes.get_mut(parent_id) {
                node.parents.push(parent_id.to_owned());
                parent.children.push(node_id.clone());
            }
        } else {
            self.root_ids.push(node_id.clone());
        }

        self.nodes.insert(node_id.clone(), node);
        node_id
    }

    /// Return a node by id.
    #[must_use]
    pub fn get_node(&self, node_id: &str) -> Option<&MerkleDagNode> {
        self.nodes.get(node_id)
    }

    /// Return all nodes (unordered).
    #[must_use]
    pub fn nodes(&self) -> Vec<&MerkleDagNode> {
        self.nodes.values().collect()
    }

    /// Serialize to a deterministic representation.
    #[must_use]
    pub fn serialize(&self) -> MerkleDagSerialized {
        let nodes = self
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), node.clone()))
            .collect();
        let mut root_ids = self.root_ids.clone();
        root_ids.sort();
        MerkleDagSerialized { nodes, root_ids }
    }

    /// Deserialize from a snapshot payload.
    #[must_use]
    pub fn deserialize(data: MerkleDagSerialized) -> Self {
        let mut dag = Self::new();
        dag.nodes = data.nodes.into_iter().collect();
        let mut root_ids = data.root_ids;
        root_ids.sort();
        dag.root_ids = root_ids;
        dag
    }

    /// Compare two DAGs and return added/removed/modified node ids.
    #[must_use]
    pub fn compare(left: &Self, right: &Self) -> MerkleDagDiff {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for key in right.nodes.keys() {
            if !left.nodes.contains_key(key) {
                added.push(key.clone());
            }
        }

        for key in left.nodes.keys() {
            if !right.nodes.contains_key(key) {
                removed.push(key.clone());
            }
        }

        for (key, left_node) in &left.nodes {
            if let Some(right_node) = right.nodes.get(key)
                && left_node.data != right_node.data
            {
                modified.push(key.clone());
            }
        }

        added.sort();
        removed.sort();
        modified.sort();

        MerkleDagDiff {
            added,
            removed,
            modified,
        }
    }
}

fn hash_data(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_serialization_is_deterministic() {
        let mut dag = MerkleDag::new();
        let root = dag.add_node("root:abc", None);
        dag.add_node("file:a", Some(&root));
        dag.add_node("file:b", Some(&root));

        let first = dag.serialize();
        let second = dag.serialize();
        assert_eq!(first, second);
    }

    #[test]
    fn dag_compare_detects_changes() {
        let mut left = MerkleDag::new();
        let root_left = left.add_node("root:abc", None);
        left.add_node("file:a", Some(&root_left));

        let mut right = MerkleDag::new();
        let root_right = right.add_node("root:def", None);
        right.add_node("file:a", Some(&root_right));
        right.add_node("file:b", Some(&root_right));

        let diff = MerkleDag::compare(&left, &right);
        assert!(!diff.is_empty());
        assert!(diff.added.iter().any(|id| id == &root_right));
    }
}
