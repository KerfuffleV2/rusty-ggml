# rusty-ggml

GGML bindings  that aim to be idiomatic Rust rather than directly corresponding to the C/C++ interface.

## GG-what?

See:

1. https://github.com/ggerganov/ggml/
2. https://github.com/KerfuffleV2/ggml-sys-bleedingedge ( https://crates.io/crates/ggml-sys-bleedingedge )

## WIP

Not suitable for general use. Consider this to be pre-alpha code.

**`v0.0.8` Warning**: Keeping this in sync with recent GGML changes has lagged. It compiles and seems to work but there might be weird stuff I haven't caught.

**Note**: There are special considerations when using GPU features like `cublas`, `hipblas`. See the `ggml-sys-bleedingedge` repo or crate documentation for more information

Example usage: https://github.com/KerfuffleV2/smolrsrwkv/blob/189915ec68b28d057b440f75803d3d056e150733/smolrwkv/src/ggml/graph.rs

## Related

For your token sampling needs see https://github.com/KerfuffleV2/llm-samplers ( https://crates.io/crates/llm-samplers )
