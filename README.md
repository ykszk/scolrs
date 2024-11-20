scoliosis


# Python binding

```console
cd crates/pyscol
maturin develop --release
RUST_LOG=info python3 -m unittest discover tests/
```

# Development

## Test
Set `TEST_OUTPUT_DIR` env var to the directory where you want to store the test output.
