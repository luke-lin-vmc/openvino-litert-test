# About this repo
This repo is used to document our testing steps for LiteRT (formerly TensorFlow Lite) and LiteRT-LM with the OpenVINO backend.

![chat_on_npu](./log/chat_on_npu.gif)

# Quick Steps
## Installation
Install and upgrade `litert-lm` to the latest version.
```
pip install --upgrade litert-lm
```
Install and upgrade `openvino` to the latest version.
```
pip install --upgrade openvino
```
Log file [installation.log](./log/installation.log) is provided for reference.

## Chat
Run below commands to download Gemma4 and run the model
### Run on CPU
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm ^
--backend=cpu ^
--prompt="What is OpenVINO?"
```
### Run on GPU
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm ^
--backend=gpu ^
--prompt="What is OpenVINO?"
```
### Run on NPU (Lunar Lake)
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it_intel_LNL.litertlm ^
--backend=npu ^
--prompt="What is OpenVINO?"
```
* Please note the model name becomes `gemma-4-E2B-it_intel_LNL.litertlm`
### Run on NPU (Panther Lake)
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it_intel_PTL.litertlm ^
--backend=npu ^
--prompt="What is OpenVINO?"
```
* Please note the model name becomes `gemma-4-E2B-it_intel_PTL.litertlm`
### Note
* There is no need to worry about models being re-downloaded, as they will be cached under `%USERPROFILE%\.cache\huggingface\hub\` after they are downloaded once
* Or you may manually download models from [HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)
### Sample log
```
(python313_venv) C:\Users\lukelin1\Downloads>litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it_intel_LNL.litertlm ^
--backend=npu ^
--prompt="What is the capital of France?"
Downloading gemma-4-E2B-it_intel_LNL.litertlm from litert-community/gemma-4-E2B-it-litert-lm...
The capital of France is **Paris**.
```
Log file [chat.log](./log/chat.log) is provided for reference.

## Audio Transcription and Translation
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm ^
--backend=gpu ^
--prompt="transcribe the audio then translate it to Chinese" ^
--attachment "how_are_you_doing_today.wav" ^
--audio-backend=cpu
```
* Audio sample [```how_are_you_doing_today.wav```](./sample/how_are_you_doing_today.wav) is attached ([source](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav))
* Supported backend are `cpu` and `gpu`
* Supported audio-backend is `cpu`
### Sample log
```
(python313_venv) C:\Users\lukelin1\Downloads>litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --backend=gpu --prompt="transcribe the audio then translate it to Chinese" --attachment "how_are_you_doing_today.wav" --audio-backend=cpu
Downloading gemma-4-E2B-it.litertlm from litert-community/gemma-4-E2B-it-litert-lm...
How are you doing today?
今天你怎么样？
```
Log file [transcribe_translate_audio.log](./log/transcribe_translate_audio.log) is provided for reference.

## Image Description and Translation
```
litert-lm run ^
--from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm ^
--backend=gpu ^
--prompt="describe and translate the picture" ^
--attachment "image_cs.jpg" ^
--vision-backend=gpu
```
* Image sample [```image_cs.jpg```](./sample/image_cs.jpg) is attached. It contains a traffic sign written in Czech characters ([`source`](https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg))
* Supported backend are `cpu` and `gpu`
* Supported vision-backend are `cpu` and `gpu`
### Sample log
**Input**
![Input](./sample/image_cs.jpg)
**Output**
```
**Text on the sign (in Czech):**
1.  **PĚŠÍ ZÓNA** (at the top)
2.  **ZÁSOBOVÁNÍ** (in the middle section)
3.  **IZS CBS V ZÁSAHU** (below the middle section)
4.  **0 - 24 h** (at the bottom)

**Graphic:**
In the center of the sign is a blue circular graphic depicting a stylized illustration of a person (likely an adult) holding the hand of a child, suggesting a pedestrian area.

**Setting:**
The sign is mounted outdoors, likely on a pole. In the background, there are trees and some buildings visible, suggesting an urban or suburban street setting.

### Translation

Here is the translation of the Czech text into English:

*   **PĚŠÍ ZÓNA:** Pedestrian Zone (or Pedestrian Area)
*   **ZÁSOBOVÁNÍ:** (This word is slightly ambiguous without more context, but in the context of traffic signs, it might relate to "supply," "provision," or perhaps be part of a specific local regulation. Given the context of the other text, it might be a specific local term.)
*   **IZS CBS V ZÁSAHU:** (This appears to be an acronym or specific regulatory instruction. Without knowing the local jurisdiction, a direct, precise translation is difficult, but it likely refers to specific rules or services related to the pedestrian zone.)
*   **0 - 24 h:** 0 - 24 hours (indicating the time frame for the regulation)
```
Log file [describe_translate_image.log](./log/describe_translate_image.log) is provided for reference.


# Reference
* [OpenVINO™ backend for LiteRT: Optimize NPU performance on Intel® Core™ Ultra processors](https://www.intel.com/content/www/us/en/developer/articles/community/litert-unlocks-core-ultra-npu-performance-for-aipc.html)
* [LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/cli)


# Future Works
* Try classical models using LiteRT on Intel NPU, for both JIT and AOT model
<br>https://ai.google.dev/edge/litert/next/intel</br>
* Try LiteRT-LM with Function Calling/Tools
<br>https://ai.google.dev/edge/litert-lm/cli#function_calling_tools</br>
