use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static REUSING_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS))
    .run_async(|ctx| {
        let data = vec![0u32; 1024];
        let input = pollster::block_on(execute_gpu(&data));

        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input).await }
    });

async fn assert_execute_gpu(device: &wgpu::Device, queue: &wgpu::Queue, input: &[u32]) {
    assert_eq!(
        input,
        vec![9; 1024],
        "Test failed: Expected all 1s after multiple passes."
    );
}
