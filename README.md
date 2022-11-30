# WasmEdge WASI-NN Benchmark

In this repository, we compare the old [`wasmedge-tensorflow`](https://github.com/second-state/WasmEdge-tensorflow-tools) and the [`wasi-nn`](https://wasmedge.org/book/en/write_wasm/rust/wasinn.html) for inferring the TensorFlow-Lite models.

## Prerequisites

WasmEdge Installation (On Ubuntu 20.04):

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -e tf,image
source $HOME/.wasmedge/env
```

The latest wasi-nn plug-in (WasmEdge 0.11.2) has an [issue](https://github.com/WasmEdge/WasmEdge/pull/2135). So please build the `manylinux2014_x86_64` of plug-in.

```bash
git clone https://github.com/WasmEdge/WasmEdge.git && cd WasmEdge
docker pull wasmedge/wasmedge:manylinux2014_x86_64
docker run -it --rm -v $(pwd):/root/$(basename $(pwd)) wasmedge/wasmedge:manylinux2014_x86_64
# In docker
cd root/WasmEdge
./utils/docker/build-manylinux.sh --release -DWASMEDGE_PLUGIN_WASI_NN_BACKEND=TensorflowLite -DWASMEDGE_BUILD_TOOLS=Off
exit
```

And copy the plug-in into the installation folder:

```bash
cp build/plugins/wasi_nn/libwasmedgePluginWasiNN.so ~/.wasmedge/plugin/
```

## Inferring the model with WasmEdge-Tensorflow and WasmEdge-Image

You can build the WASM file by yourself or use the prebuilt one in the folder:

```bash
cd <path/to/this/repo>
cd wasmedge-tf-wasmedge-image
cargo build --release --target=wasm32-wasi
# The output file is at `target/wasm32-wasi/release/wasmedge-tf-wasmedge-image.wasm`
```

Run the example by `wasmedge-tensorflow`:

```bash
$ cd <path/to/this/repo>
$ wasmedge-tensorflow-lite --dir .:. wasmedge-tf-wasmedge-image/wasmedge-tf-wasmedge-image.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 61.70ms
time to read and resize image: 37.78ms
time to infer and get output: 143.95ms
time to search the highest probability: 330.80µs
166 : 0.87058824
```

And execute in AOT mode:

```bash
$ wasmedgec wasmedge-tf-wasmedge-image/wasmedge-tf-wasmedge-image.wasm wasmedge-tf-wasmedge-image/wasmedge-tf-wasmedge-image_aot.wasm
$ wasmedge-tensorflow-lite --dir .:. wasmedge-tf-wasmedge-image/wasmedge-tf-wasmedge-image_aot.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 3.27ms
time to read and resize image: 31.57ms
time to infer and get output: 53.96ms
time to search the highest probability: 7.50µs
166 : 0.87058824
```

## Inferring the model with WasmEdge-Tensorflow with loading image in rust

You can build the WASM file by yourself or use the prebuilt one in the folder:

```bash
cd <path/to/this/repo>
cd wasmedge-tf-rust-image
cargo build --release --target=wasm32-wasi
# The output file is at `target/wasm32-wasi/release/wasmedge-tf-rust-image.wasm`
```

Run the example by `wasmedge-tensorflow`:

```bash
$ cd <path/to/this/repo>
$ wasmedge-tensorflow-lite --dir .:. wasmedge-tf-rust-image/wasmedge-tf-rust-image.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 61.68ms
time to read and resize image: 9.68s
time to infer and get output: 141.41ms
time to search the highest probability: 381.20µs
166 : 0.7411765
```

And execute in AOT mode:

```bash
$ wasmedgec wasmedge-tf-rust-image/wasmedge-tf-rust-image.wasm wasmedge-tf-rust-image/wasmedge-tf-rust-image_aot.wasm
$ wasmedge-tensorflow-lite --dir .:. wasmedge-tf-rust-image/wasmedge-tf-rust-image_aot.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 3.58ms
time to read and resize image: 34.93ms
time to infer and get output: 52.72ms
time to search the highest probability: 7.50µs
166 : 0.7411765
```

## Inferring the model with wasi-nn plug-in with loading image in rust

You can build the WASM file by yourself or use the prebuilt one in the folder:

```bash
cd <path/to/this/repo>
cd wasi-nn-rust-image
cargo build --release --target=wasm32-wasi
# The output file is at `target/wasm32-wasi/release/wasi-nn-rust-image.wasm`
```

Run the example by `wasmedge`:

```bash
$ cd <path/to/this/repo>
$ wasmedge --dir .:. wasi-nn-rust-image/wasi-nn-rust-image.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 60.04ms
time to read and resize image: 9.72s
time to infer and get output: 55.05ms
time to search the highest probability: 577.60µs
166 : 0.7411765
```

And execute in AOT mode:

```bash
$ wasmedgec wasi-nn-rust-image/wasi-nn-rust-image.wasm wasi-nn-rust-image/wasi-nn-rust-image_aot.wasm
$ wasmedge --dir .:. wasi-nn-rust-image/wasi-nn-rust-image_aot.wasm lite-model_aiy_vision_classifier_birds_V1_3.tflite bird.jpg
time to read model: 3.48ms
time to read and resize image: 35.90ms
time to infer and get output: 52.77ms
time to search the highest probability: 7.70µs
166 : 0.7411765
```

## Compare

| Case                                           | Interpreter Time | AOT Time |
| ---------------------------------------------- | ---------------- | -------- |
| WasmEdge-Tensorflow with WasmEdge-Image        | 0.266s           | 0.106s   |
| WasmEdge-Tensorflow with rust image processing | 9.956s           | 0.115s   |
| WASI-NN plug-in with rust image processing     | 9.833s           | 0.123s   |
