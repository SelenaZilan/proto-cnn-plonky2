#![allow(clippy::missing_panics_doc)]

mod allocator;

use allocator::Allocator;

#[cfg(not(feature = "mimalloc"))]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::StdAlloc> = allocator::new_std();

#[cfg(feature = "mimalloc")]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::MiMalloc> = allocator::new_mimalloc();
pub mod smalliris_real_zk;
pub mod smalliris_real_zk_recursive;
