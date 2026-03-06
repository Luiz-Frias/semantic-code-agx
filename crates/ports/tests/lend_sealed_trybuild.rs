//! Trybuild UI checks for sealed lending port shims.

#[test]
fn lending_traits_are_sealed() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/lend_bridge_pass.rs");
    tests.pass("tests/ui/vectordb_lend_bridge_pass.rs");
    tests.compile_fail("tests/ui/embedding_lend_external_impl.rs");
    tests.compile_fail("tests/ui/vectordb_lend_external_impl.rs");
}
