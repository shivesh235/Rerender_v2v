# **Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation**

## **Overview**
This project implements a novel zero-shot, text-guided video-to-video translation framework built on diffusion models and ControlNet. The system enhances existing methods by introducing features like **keyframe super-resolution**, **night-light enhancement**, and **audio adaptation** for the generated videos. The framework ensures **global and local temporal consistency** in rendering and provides high-quality outputs without retraining.

---

## **Key Features**
1. **Hierarchical Cross-Frame Constraints**:
   - Enforces **shape**, **texture**, and **color coherence** at different stages of the diffusion sampling process.
   - Uses optical flow-based dense constraints to maintain temporal consistency.

2. **Super-Resolution and Low-Light Enhancement**:
   - **Super-Resolution**: Enhances the resolution of keyframes using the NinaSR model.
   - **Night-Light Enhancement**: Enhances visibility in low-light frames using the MIRNet model.

3. **Audio Integration**:
   - Adds or adapts sound to generated videos, improving their realism and usability.

4. **Text-Guided Generation**:
   - Input textual prompts drive the style and content of the generated video frames.

5. **UI for Interactive Video Generation**:
   - Gradio-based interface for easy configuration, execution, and visualization.

6. **Extensibility**:
   - Supports ControlNet, GMFlow for optical flow, and compatibility with LoRA and ControlNet for customization.

---

## **Pipeline**
1. **Input Preprocessing**:
   - Extracts frames from the input video, resizes, and crops them to fit the specified resolution.

2. **Key Frame Translation**:
   - Generates keyframes using diffusion models guided by ControlNet and text prompts.

3. **Full Video Translation**:
   - Uses optical flow to propagate keyframes across video frames with **temporal-aware patch matching** and blending.

4. **Super-Resolution & Night-Light Enhancement**:
   - Super-resolves keyframes and enhances low-light conditions as needed.

5. **Audio Adaptation**:
   - Syncs or integrates audio into the final output.

---

## **Requirements**
### **Dependencies**:
- **Python 3.8+**
- Core libraries: `torch`, `torchvision`, `einops`, `numpy`, `PIL`, `cv2`
- ControlNet: Pre-trained models for HED and Canny
- Optical Flow: GMFlow (`gmflow_sintel-0c07dcb3.pth`)
- Diffusion Model: ControlLDM, FreeU modifications
- Low-Light Enhancement: MIRNet
- Super-Resolution: NinaSR
- UI: `gradio`

### **Installation**:
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required models and place them in the `models/` directory:
   - GMFlow: `gmflow_sintel-0c07dcb3.pth`
   - ControlNet HED: `control_sd15_hed.pth`
   - ControlNet Canny: `control_sd15_canny.pth`
   - VAE weights: `vae-ft-mse-840000-ema-pruned.ckpt`

---

## **Usage**
### **Command-Line Inference**:
1. **Run Inference**:
   ```bash
   python inference.py --input <path_to_input_video> --output <path_to_output_video> --prompt <text_prompt>
   ```
2. **Options**:
   - `--key_video_path`: Path to save keyframe video.
   - `-one`: Generate only the first frame.
   - `--n_proc`: Number of processes for parallel blending.

### **Gradio-Based UI**:
Launch the Gradio interface for an interactive workflow:
```bash
python webUI.py
```
#### **UI Features**:
- Upload video and provide a text prompt.
- Choose advanced options (e.g., control strength, resolution, denoising strength).
- Apply super-resolution or night-light enhancement on keyframes.

![webui](https://github.com/shivesh235/Rerender_v2v/blob/main/webui.jpg)
---

## **File Structure**
- `inference.py`: Script for running the video-to-video translation pipeline.
- `webUI.py`: Gradio interface for interactive video generation.
- `src/`: Core modules for diffusion models, attention control, and blending.
- `deps/`: Dependencies for ControlNet and GMFlow.

---

## **Models Used**
1. **ControlNet**: Guides the generation with edge maps from Canny or HED detectors.
2. **GMFlow**: Optical flow estimation for temporal consistency.
3. **NinaSR**: Super-resolution model for enhancing keyframes.
4. **MIRNet**: Model for low-light video enhancement.
5. **AudioLDM:** Model for sound effects generation.

---

## **Examples**
### **Input**:
- Video: Low-resolution, low-light frames.
- Prompt: *"Transform the video into a Van Gogh painting style"*

### **Output**:
- High-resolution video with enhanced brightness and texture, consistent across frames, and accompanied by adapted audio.

---

## **Future Work**
1. Extend support for 3D video generation.
2. Integrate additional audio synthesis models.
3. Optimize for real-time performance.

---
