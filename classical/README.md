# About this folder
This folder is used to document our testing steps for running classical (vision, audio, and NLP) models on LiteRT (formerly TensorFlow Lite) with the OpenVINO backend.

The model can be executed with the following two modes
* just-in-time (JIT)
* ahead-of-time (AOT) --- compilation of a provided .tflite model.

# Steps

### This example is to run MobileNet_v2 from [ImageNet LiteRT end-to-end sample](https://github.com/google-ai-edge/litert-samples/tree/main/end_to_end/imagenet)

## Export model (Linux only)
Model export is only supported on Linux as the required package [litert-torch](https://github.com/google-ai-edge/litert-torch) requires Linux

Run below commands to generate `mobilenet_v2.tflite` and quantized `mobilenet_v2.int8.tflite`
```
pip install uv
uv run main.py convert --arch mobilenet_v2
uv run main.py convert --arch mobilenet_v2 --output mobilenet_v2.int8.tflite --quantize
```
Log file [export.log](./export.log) is provided for reference.

## Run model
### Download sample image
```
curl -o coco.jpg https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg
```
### Download label files
The script requires ImageNet label metadata to map model outputs to human-readable names.
```
curl -sSL -o imagenet_lsvrc_2015_synsets.txt https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
```
```
curl -sSL -o imagenet_metadata.txt https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_metadata.txt
```
### Install required packages

Install OpenVINO's `ai-edge-litert` and `ai-edge-litert-sdk`

Reference: https://ai.google.dev/edge/litert/next/intel

```
pip install --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly ai-edge-litert-nightly ai-edge-litert-sdk-intel-nightly
```
Install other required packages
```
pip install -r requirements.txt
```
Verify installation
```
python check.py
```
Expected output
```
Backend: intel_openvino
Dispatch: C:\Python\python313_venv\Lib\site-packages\ai_edge_litert\vendors\intel_openvino\dispatch
OpenVINO: 2026.2.0-21820-9a25caa5a15
SDK libs: ['openvino_intel_npu_compiler.dll', 'openvino_intel_npu_compiler_loader.dll']
Available devices: ['CPU', 'GPU', 'NPU']
```
Log file [install.log](./install.log) is provided for reference.

### Run
Input below command
```
python main.py --model mobilenet_v2.tflite --image coco.jpg
```
* You can also test the quantized `mobilenet_v2.int8.tflite` model

**Input**
<p><img src="./coco.jpg" width="400" alt="Czech traffic signs"></p>

**Expected Result**
```
1: n02099267 flat-coated retriever (0.443499)
2: n02099712 Labrador retriever (0.324956)
3: n02093256 Staffordshire bullterrier, Staffordshire bull terrier (0.079632)
4: n02109047 Great Dane (0.044876)
5: n02111277 Newfoundland, Newfoundland dog (0.027100)
```
Log file [run.log](./run.log) is provided for reference.

### Know issue
Fail to run on NPU, log below. Still WIP
```
WARNING: [litert/runtime/accelerators/npu_registry.cc:34] NPU accelerator could not be loaded and registered: kLiteRtStatusErrorInvalidArgument.
```
# Reference
* [OpenVINO™ backend for LiteRT: Optimize NPU performance on Intel® Core™ Ultra processors](https://www.intel.com/content/www/us/en/developer/articles/community/litert-unlocks-core-ultra-npu-performance-for-aipc.html)
* [Intel NPU (OpenVino) with LiteRT](https://ai.google.dev/edge/litert/next/intel)
