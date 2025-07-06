use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static TWO_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::BUFFER_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits {
              
              
                ..Default::default()
            }),
    )
    .run_async(|ctx| {
       
       let numbers = vec![0u32; 1024];

        let input = pollster::block_on(execute_gpu(&numbers));

      
        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input).await }
    });

async fn assert_execute_gpu(device: &wgpu::Device, queue: &wgpu::Queue, input: &[u32]) {

      assert_eq!(
            input,
            vec![9; 1024],
            "Test failed: Expected all 1s after multiple passes."
        );

    
}
