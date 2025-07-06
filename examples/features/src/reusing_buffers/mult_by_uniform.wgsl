@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<uniform> factor: u32;

@compute @workgroup_size(64) // Matches the WORKGROUP_SIZE constant in Rust
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index < arrayLength(&data) {
        data[index] = data[index] * factor;
    }
}
