//! Loom model checks for one-time initialization invariants.

#![cfg(feature = "loom-tests")]

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

#[test]
fn once_initializer_runs_exactly_once_under_race() {
    loom::model(|| {
        let init_guard = Arc::new(Mutex::new(()));
        let initialized = Arc::new(loom::sync::atomic::AtomicBool::new(false));
        let init_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..2 {
            let init_guard_task = Arc::clone(&init_guard);
            let initialized_task = Arc::clone(&initialized);
            let init_count_task = Arc::clone(&init_count);
            handles.push(thread::spawn(move || {
                if !initialized_task.load(Ordering::Acquire) {
                    let _lock = init_guard_task.lock().expect("lock should not be poisoned");
                    if !initialized_task.load(Ordering::Relaxed) {
                        init_count_task.fetch_add(1, Ordering::SeqCst);
                        initialized_task.store(true, Ordering::Release);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().expect("loom thread must join cleanly");
        }

        assert_eq!(init_count.load(Ordering::SeqCst), 1);
        assert!(initialized.load(Ordering::Acquire));
    });
}
