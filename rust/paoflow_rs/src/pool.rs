//! The rayon pool shared by every kernel.
//!
//! `PAOFLOW_RS_THREADS` (legacy `PAOFLOW_ACBN0_THREADS`) sizes a dedicated pool; PAOFLOW
//! exports it from this rank's share of the node under `mpirun`
//! (`PAOFLOW.utils.parallel_resources.configure_native_threads`), so that N ranks on a node
//! do not each start a pool over every core.  Without the variable the global rayon pool is
//! used (which honours `RAYON_NUM_THREADS`).

use std::sync::OnceLock;

fn custom() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        std::env::var("PAOFLOW_RS_THREADS")
            .or_else(|_| std::env::var("PAOFLOW_ACBN0_THREADS"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .expect("failed to build PAOFLOW_RS_THREADS rayon pool")
            })
    })
    .as_ref()
}

/// Run `f` inside the configured pool (or the global one).
pub fn install<R, F>(f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    match custom() {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Threads the kernels use.
pub fn thread_count() -> usize {
    match custom() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
