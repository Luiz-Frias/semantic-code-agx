//! Miri-targeted smoke test for ports invariants.

use semantic_code_ports::{EmbeddingVector, EmbeddingVectorFixed, SafeRelativePath};

#[test]
fn path_and_embedding_roundtrip_is_memory_safe() {
    let path = SafeRelativePath::new("src//lib.rs").expect("path should normalize");
    assert_eq!(path.as_str(), "src/lib.rs");

    let vector = EmbeddingVector::from_vec(vec![0.1, 0.2, 0.3]);
    let fixed = EmbeddingVectorFixed::<3>::try_from(vector.clone())
        .expect("fixed-size conversion should succeed");
    let restored: EmbeddingVector = fixed.into();
    assert_eq!(restored.dimension(), 3);
    assert_eq!(restored.as_slice(), &[0.1, 0.2, 0.3]);
}
