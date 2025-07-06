use log::{debug, info, trace};
use std::mem;
use wgpu::{util::DeviceExt, BufferAddress, Features};

const WORKGROUP_SIZE: u32 = 64;
const NUM_ELEMENTS: usize = 1 << 16; // 65K numbers in our 'data'

pub async fn run() {
    let numbers = vec![0u32; NUM_ELEMENTS];
    debug!(
        "Initial numbers buffer created with {} elements (all zeros).",
        NUM_ELEMENTS
    );

    // Execute the compute passes on the GPU.
    let results = execute_gpu(&numbers).await;

    // Expected result calculation:
    //   Initialise with zeros.
    //   Add one to make all 1s
    //   Multiply by 10 to make all 10s.
    //   Subtract one to result in all 9s
    assert!(
        results.into_iter().all(|v| v == 9),
        "Results not as expected! Expected all 9s after multiple passes.
    ",
    );
}

pub async fn execute_gpu(numbers: &[u32]) -> Vec<u32> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to request adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: wgpu::Limits {
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, numbers).await
}

pub async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    numbers: &[u32],
) -> Vec<u32> {
    let buffer_size = std::mem::size_of_val(numbers) as BufferAddress;
    debug!(
        "Buffer size: {} bytes for {} elements.",
        buffer_size,
        numbers.len()
    );

    // --- 1. Create Buffers ---
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(numbers),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Uniform Buffer"),
        size: mem::size_of::<u32>() as BufferAddress, // Just enough space for one u32
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- 2. Define Bind Group Layout and Create Bind Group ---
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, // Corresponds to @binding(0) in WGSL
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, // Corresponds to @binding(1) in WGSL
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    // --- 3. Load ALL Shaders & Create Compute Pipelines for Each Pass ---

    let shader_add_one = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader Add One"),
        source: wgpu::ShaderSource::Wgsl(include_str!("add_one.wgsl").into()),
    });
    let pipeline_add_one = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pipeline Add One"),
        layout: Some(
            // Create a pipeline layout with our bind group layout
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout], // Our defined layout
                push_constant_ranges: &[],
            }),
        ),
        module: &shader_add_one,
        entry_point: Some("main"), // You could use just one shader, and use different entry points!
        compilation_options: Default::default(),
        cache: None,
    });

    let shader_multiply = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader Multiply"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mult_by_uniform.wgsl").into()),
    });
    let pipeline_multiply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pipeline Multiply"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: &shader_multiply,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // NOTE: We're re-using the same `bind_group_layout`, and just like regular Rust,
    // you're using a `&` reference to this GPU memory.

    let shader_subtract = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader Subtract"),
        source: wgpu::ShaderSource::Wgsl(include_str!("subtract_one.wgsl").into()),
    });
    let pipeline_subtract = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pipeline Subtract"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: &shader_subtract,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    //
    //  NOTE: each shader has its OWN pipeline, however you could create fewer ShaderModules, with more entry points and then reuse them.
    //

    // Number of workgroups needed to cover all elements.
    let workgroups_x = (NUM_ELEMENTS as u32).div_ceil(WORKGROUP_SIZE);

    // --- 4. Record Commands for Multiple Compute Passes with Synchronization ---

    // Pass 1: Add 1
    {
        info!("Beginning Pass 1: Add One.");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pass 1 (Add One)"),
        });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass 1 (Add One)"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline_add_one);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, 1, 1);
        drop(cpass);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait).unwrap();
        info!("Pass 1 complete. Data should now be [1, 1, ...].");
    }

    // Pass 2: Multiply by 2
    {
        info!("Beginning Pass 2: Multiply by 10.");
        // Create a new encoder for this pass.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pass 2 (Multiply by Uniform)"),
        });

        const MUL_VAL: u32 = 10;

        // This write happens directly on the queue, before the encoder's commands are submitted.
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[MUL_VAL]));
        info!("Uniform buffer updated to 10.");

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass 2 (Multiply by Uniform)"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline_multiply);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, 1, 1);
        drop(cpass);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait).unwrap();
        info!("Pass 2 complete. Data should now be [2, 2, ...].");
    }

    // Pass 3: Subtract 1
    {
        info!("Beginning Pass 3: Subtract 1.");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pass 3 (Subtract Uniform)"),
        });

        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[1u32]));
        info!("Uniform buffer's value to sub updated to 1.");

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass 3 (Subtract Uniform)"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline_subtract);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, 1, 1);
        drop(cpass);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait).unwrap();
        info!("Pass 3 complete. Data should now be [9, 9, ...].");
    }

    //
    // NOTE: We've submitted and polled for all the jobs we made so they're going to return sequentially,
    // however you can submit submit multiple jobs to the queue, see https://docs.rs/wgpu/latest/wgpu/struct.Queue.html#method.submit
    //

    // --- 5. Copy Results Back to Staging Buffer ---
    {
        debug!("Copying results from storage buffer to staging buffer.");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Copy to Staging"),
        });
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish())); // Submit copy commands
        trace!("Copy commands submitted.");
    }

    // --- 6. Map and Read Results ---
    debug!("Mapping staging buffer for read.");
    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| match result {
        Ok(_) => info!("Staging buffer mapped successfully."),
        Err(e) => log::error!("Failed to map staging buffer: {:?}", e),
    });

    device.poll(wgpu::PollType::Wait).unwrap();
    trace!("Device polled, mapping should be complete.");

    // Get the mapped range and cast it to a `Vec<u32>`.
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    info!(
        "Results read from mapped buffer. First 10 elements: {:?}",
        &result[0..std::cmp::min(10, result.len())]
    );

    drop(mapped);
    staging_buffer.unmap();
    trace!("Staging buffer unmapped.");

    result
}

/// Main function that initializes logging and runs the asynchronous `run` function.
pub fn main() {
    env_logger::init();
    trace!("Application started. Env logger initialized.");
    pollster::block_on(run());
}
