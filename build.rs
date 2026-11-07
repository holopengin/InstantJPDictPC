// The final link needs the OpenMP runtime: the pinned ncnn fork (built with
// OpenMP on, same as the mobile build) references GOMP_* symbols, and the
// native objects ship inside `jpdict_core`'s rlib. `cargo:rustc-link-arg`
// only applies to the crate being linked, so it must be emitted here — the
// core crate's own build script cannot add link args to this binary.
fn main() {
    println!("cargo:rustc-link-arg=-fopenmp");
}
