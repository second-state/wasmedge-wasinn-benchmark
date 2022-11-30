use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use wasi_nn;
use image;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let image_name: &str = &args[2];

    let mut now = Instant::now();

    let mut file_mod = File::open(model_name).unwrap();
    let mut mod_buf = Vec::new();
    file_mod.read_to_end(&mut mod_buf).unwrap();

    println!("time to read model: {:.2?}", now.elapsed());
    now = Instant::now();

    // Use rust crate to load and resize image
    let flat_img = 
        image::io::Reader::open(image_name).unwrap()
        .decode().unwrap()
        .resize_exact(224, 224, image::imageops::Triangle)
        .into_rgb8().as_raw().to_vec();
    
    println!("time to read and resize image: {:.2?}", now.elapsed());
    now = Instant::now();

    let mut res_vec = vec![0u8; 965];
    unsafe{
        let graph = 
            wasi_nn::load(
                &[&mod_buf],
                4, // encoding for tflite: wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE
                wasi_nn::EXECUTION_TARGET_CPU,
            ).unwrap();
        let context = wasi_nn::init_execution_context(graph).unwrap();
        let tensor = wasi_nn::Tensor {
            dimensions: &[1, 224, 224, 3],
            r#type: wasi_nn::TENSOR_TYPE_U8,
            data: &flat_img,
        };
        wasi_nn::set_input(context, 0, tensor).unwrap();
        wasi_nn::compute(context).unwrap();
        wasi_nn::get_output(
            context,
            0,
            &mut res_vec[..] as *mut [u8] as *mut u8,
            res_vec.len().try_into().unwrap(),
        ).unwrap();
    }

    println!("time to infer and get output: {:.2?}", now.elapsed());
    now = Instant::now();

    let mut i = 0;
    let mut max_index: i32 = -1;
    let mut max_value: u8 = 0;
    while i < res_vec.len() {
        let cur = res_vec[i];
        if cur > max_value {
            max_value = cur;
            max_index = i as i32;
        }
        i += 1;
    }
    
    println!("time to search the highest probability: {:.2?}", now.elapsed());

    println!("{} : {}", max_index, max_value as f32 / 255.0);
}
