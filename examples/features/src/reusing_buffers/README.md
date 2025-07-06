# reusing buffers


Demonstrates how to reuse a `wgpu::Buffer` via the `BindGroup` it is instantiated with to avoid reallocating GPU memory.

- Creates an input of `[0,0,0,0,.......]`s
- Puts those `0s` onto the GPU.
- Adds `1` to all of them, with a compute pass.
- Multiplies them all by `10`, with a second compute pass, that uses the same `BindGroup` (i.e same GPU memory)
- Removes `1` from all of them (leaving an array of `9`s), with a third pass, _again_ using the same `BindGroup`


## To Run

```sh
# linux/mac
RUST_LOG=wgpu_examples::reusing_buffers=info cargo run -r --bin wgpu-examples -- reusing_buffers

# windows (Powershell)
$env:WGPU_BACKEND="Vulkan"; $env:RUST_LOG="wgpu_examples::reusing_buffers=info"; cargo run -r --bin wgpu-examples -- reusing_buffers
```
