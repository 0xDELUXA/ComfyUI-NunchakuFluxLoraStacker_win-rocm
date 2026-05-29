# ControlAltAI-Nodes-fixed-Python3.13

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../nodes/controlaltai/controlalttai.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

这是原始 ControlAltAI-Nodes 的 Python 3.13 兼容版本。

## 概述

由于原始的 ControlAltAI-Nodes 在 Python 3.13 下停止工作，因此创建了此版本。在此版本中修改了所有节点的名称，以避免与名称相似的节点发生冲突。

## 为什么在 Python 3.13 下需要重命名

Python 3.13 缩紧了关于模块加载和缓存验证的若干内部机制。ComfyUI 通过让每个扩展导入其模块并向两个全局字典中追加条目来发现自定义节点：`NODE_CLASS_MAPPINGS`（类名 → 实现）和 `NODE_DISPLAY_NAME_MAPPINGS`（类名 → UI 标签）。一旦类名或显示名称与已注册的条目冲突，后导入的模块要么会覆盖现有条目，要么会直接跳过注册，导致该节点在 UI 中丢失。在较早的 Python 版本中，这种冲突通常是无害的；但在 3.13 中，这会导致节点无提示地消失。

主要原因包括：

- **更严格的 importlib 重用**：Python 3.13 会积极地重用模块规格（module specs）。当两个模块共享同一个 `module.__spec__` 时，第二次导入会短路，其主体内容永远不会执行，因此本应注册节点的调用也就无法运行。

- **字节码缓存强制验证**：现在过期的 `.pyc` 文件会被立即拒绝。如果两个文件保持相同的路径（例如 `flux_resolution_cal_node.py`），则一个版本的缓存字节码会被重用于另一个版本。当时间戳或哈希值不一致时，导入就会失败，导致节点不显示。

- **ComfyUI 中的全局键冲突**：ComfyUI 目录构建器假定显示名称是唯一的。多个“Flux Resolution Calc”条目会使 UI 目录的生成产生混淆，即使导入成功，重复的条目也会被丢弃。

为了保证每个节点在 Python 3.13 更严格的语义下正常工作，该分支对模块、类和显示标签进行了重命名（例如，`FluxResolutionNode` → `MegapixelCalculatorNode`，在 UI 中显示为 `ControlAltAI: Megapixel Calculator`）。同时更新了文件名，使缓存的字节码和 `module.__spec__` 条目不再发生冲突。顶层的 `__init__.py` 也被重写，以支持相对和绝对导入、输出诊断日志，并避免意外的重复注册。通过这些更改，Python 3.13 会将每个节点模块视为独立的模块，使 ComfyUI 能够可靠地加载它们。

## 节点名称变更

所有节点的类名和显示名称均已相对原始仓库进行了修改。以下是完整的修改列表：

## 节点类名与显示名称变更对照表

| 原始节点类名 | 修改后节点类名 | 原始显示名称 | 修改后显示名称 |
|-------------------------|------------------------|----------------------|---------------------|
| FluxResolutionNode | MegapixelCalculatorNode | Flux Resolution Calc | ControlAltAI: Megapixel Calculator |
| FluxSampler | FluxSampler | Flux Sampler | ControlAltAI: Advanced Sampler |
| FluxUnionControlNetApply | FluxUnionControlNetApply | Flux Union ControlNet Apply | ControlAltAI: Union ControlNet Apply |
| BooleanBasic | BooleanBasic | Boolean Basic | ControlAltAI: Boolean Basic |
| BooleanReverse | BooleanReverse | Boolean Reverse | ControlAltAI: Boolean Reverse |
| GetImageSizeRatio | GetImageSizeRatio | Get Image Size Ratio | ControlAltAI: Get Image Size Ratio |
| IntegerSettings | IntegerSettings | Integer Settings | ControlAltAI: Integer Settings |
| IntegerSettingsAdvanced | IntegerSettingsAdvanced | Integer Settings Advanced | ControlAltAI: Integer Settings Advanced |
| PerturbationTexture | PerturbationTexture | Perturbation Texture | ControlAltAI: Perturbation Texture |
| TextBridge | TextBridge | Text Bridge | ControlAltAI: Text Bridge |
| TwoWaySwitch | TwoWaySwitch | Switch (Two Way) | ControlAltAI: Switch (Two Way) |
| ThreeWaySwitch | ThreeWaySwitch | Switch (Three Way) | ControlAltAI: Switch (Three Way) |

### 关键变更摘要

1. **FluxResolutionNode → MegapixelCalculatorNode**：修改了主要节点的类名，并将文件从 `flux_resolution_cal_node.py` 重命名为 `megapixel_calculator_node.py`。

2. **所有显示名称**：现在所有节点的显示名称都带有 "ControlAltAI: " 前缀，以与原始节点进行区分。

3. **显示名称更新**：部分显示名称进行了如下更新：
   - "Flux Sampler" → "ControlAltAI: Advanced Sampler"
   - "Flux Attention Control" → "ControlAltAI: Attention Control"

## 节点列表

### 节点列表：
- Flux (Flux模型)
  - ControlAltAI: Megapixel Calculator (百万像素计算器，2025年5月更新)
  - ControlAltAI: Advanced Sampler (高级采样器)
  - ControlAltAI: Union ControlNet Apply (Union ControlNet应用)
- Logic (逻辑)
  - ControlAltAI: Boolean Basic (基础布尔值)
  - ControlAltAI: Boolean Reverse (布尔值取反)
  - ControlAltAI: Integer Settings (整数设置)
  - ControlAltAI: Integer Settings Advanced (高级整数设置，2025年6月新增)
  - ControlAltAI: Switch (Two Way) (双路开关，2025年6月新增)
  - ControlAltAI: Switch (Three Way) (三路开关，2025年6月新增)
- Image (图像)
  - ControlAltAI: Get Image Size Ratio (获取图像尺寸比例)
  - ControlAltAI: Perturbation Texture (扰动纹理，2025年6月新增)
- Utility (实用工具)
  - ControlAltAI: Text Bridge (文本桥接，2025年6月新增)

<a id="controlaltai-megapixel-calculator"></a>

### ControlAltAI: Megapixel Calculator (百万像素计算器)

ControlAltAI: Megapixel Calculator 旨在为使用 Flux 模型生成的输出确定最佳图像分辨率（Flux 模型明显更倾向于基于百万像素进行处理）。与依赖标准 SDXL 分辨率的传统方法不同，该计算器基于用户指定的百万像素输入运行。用户可以选择所需的百万像素数（范围从 0.1 到 2.0 百万像素）和纵横比。然后，计算器会提供与 Flux 模型配合使用时达到最佳性能所需的精确图像尺寸。这种方法确保了生成的图像符合针对用户需求量身定制的特定质量和尺寸要求。此外，虽然官方限制是 2.0 百万像素，但在测试过程中，我成功生成了更高分辨率的图像，这表明了该模型在适应各种图像尺寸方面的灵活性，且不会影响质量。

- **支持的百万像素**：0.1 MP - 2.5 MP（将步长更改为 0.1 可进行精细选择）
- **注意**：生成超过 1 MP 的图像可能会显得稍微模糊，但在 Flux1.Dev 模型上已成功测试过 3k+ 的分辨率。
- **自定义比例**：现在已支持自定义比例。启用或禁用自定义比例并输入任意比例（例如：4:9）。
- **预览**：预览节点仅用于直观展示比例。
- **整除底数（Divisible By）**：现在您可以选择被 8/16/32/64 整除。默认值为 64。若要获得精细微调的结果，请选择被 8 整除。对于 Flux Dev 1，建议选择被 32 或 64 整除。

<a id="controlaltai-advanced-sampler"></a>

### ControlAltAI: Advanced Sampler (高级采样器)

ControlAltAI: Advanced Sampler 节点将 CustomSamplerAdvance 节点和输入节点的功能合并到了一个简化且高效的节点中。

- **CFG 设置**：CFG 固定为 1。
- **条件输入**：仅支持正向条件（positive conditioning）。
- **兼容性**：仅包含与 Flux 模型兼容的采样器和调度器。
- **Latent 兼容性**：请仅使用 SD3 Empty Latent Image（SD3 空 Latent 图像）节点。常规的空 Latent 图像节点并不兼容。

<img src="../nodes/controlaltai/png/Advanced%20Sampler.png" width="400" alt="ControlAltAI: Advanced Sampler">

<a id="controlaltai-union-controlnet-apply"></a>

### ControlAltAI: Union ControlNet Apply (Union ControlNet 应用)

ControlAltAI: Union ControlNet Apply 节点是一个与 InstantX Union Pro ControlNet 兼容的多合一节点。它已针对 Union ControlNet 类型进行了广泛的测试，并可按预期工作。您可以组合两个 ControlNet Union 单元并获得良好的效果。不建议组合两个以上。此 ControlNet 仅在 Flux 1.Dev 模型上进行了测试。

<img src="../nodes/controlaltai/png/Union%20ControlNet%20Apply.png" width="400" alt="ControlAltAI: Union ControlNet Apply">

**推荐设置**：<br>
强度 (strength)：0.15 - 0.65。<br>
结束百分比 (end percentage)：0.200 - 0.900。

**推荐预处理器**：<br>
Canny：Canny Edge (ControlNet Aux 预处理器)。<br>
Tile：Tile (ControlNet Aux 预处理器)。<br>
Depth：Depth Anything V2 Relative (ControlNet Aux 预处理器)。<br>
Blur (注：原文Blue为拼写错误，代表Blur模糊控制)：直接输入（模糊图像）或 Tile (ControlNet Aux 预处理器)。<br>
Pose：DWPose Estimator (ControlNet Aux 预处理器)。<br>
Gray：Image Desaturate (Comfy Essentials 自定义节点)。<br>
Low Quality：直接输入。

效果展示：（不包含 Canny 和 Depth 的示例，因为这两种方式非常直观。）<br><br>
**像素级低分辨率转高分辨率**<br><br>
**照片修复**<br><br>
**游戏资源低分辨率放大**<br><br>
**模糊转清晰**<br><br>
**重着色**<br><br>

**YouTube 上的 Union ControlNet 使用教程：[视频教程](https://www.youtube.com/watch?v=4_1A5pQkJkg)**

**Shakker Labs & InstantX Flux ControlNet Union Pro 模型下载：[Hugging Face 链接](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro)**

<a id="controlaltai-get-image-size-ratio"></a>

### ControlAltAI: Get Image Size Ratio (获取图像尺寸比例)
此节点旨在获取图像的宽度、高度分辨率以及比例。该节点可以进一步连接到 ControlAltAI: Megapixel Calculator 节点。操作步骤如下：
- 右键点击 ControlAltAI: Megapixel Calculator -> Convert widget to input -> Convert custom_aspect_ratio to input。
- 将 Ratio 输出端连接到 custom_aspect_ratio 输入端。

<img src="../nodes/controlaltai/png/Get%20Image%20Size%20Ratio.png" width="400" alt="ControlAltAI: Get Image Size Ratio">

<a id="controlaltai-integer-settings"></a>

### ControlAltAI: Integer Settings (整数设置)
此节点旨在输出 1 或 2 两个整数中的一个原始值。Enable（启用） = 2，Disable（禁用） = 1。

使用场景：可以将其设置在双路开关（two-way switch）之前，从而让工作流的逻辑控制流向其中一个方向。目前，它仅支持控制两条逻辑流。未来，我们将升级该节点以支持三路或更多路的逻辑开关流。

<img src="../nodes/controlaltai/png/Integer%20Settings.png" width="400" alt="ControlAltAI: Integer Settings">

<a id="controlaltai-integer-settings-advanced"></a>

### ControlAltAI: Integer Settings Advanced (高级整数设置)
此节点旨在输出 1、2 或 3 三个整数中的一个原始值。每次只能启用一个整数输出。将此节点与新的 ControlAltAI: Switch (Three Way)（三路开关）连接即可进行逻辑控制。

<img src="../nodes/controlaltai/png/Integer%20Settings%20Advanced.png" width="400" alt="ControlAltAI: Integer Settings Advanced">

<a id="controlaltai-switch-two-way"></a>

### ControlAltAI: Switch (Two Way) (双路开关) / <a id="controlaltai-switch-three-way"></a> ControlAltAI: Switch (Three Way) (三路开关)
与仅接受单一输入类型的传统开关不同，这些开关可以接受多种输入类型，并在连接到正确的输出端时透传这些输入。现在，您可以将 ControlAltAI: Switch (Two Way) 与 ControlAltAI: Integer Settings 无缝连接，或将 ControlAltAI: Switch (Three Way) 与 ControlAltAI: Integer Settings Advanced 节点无缝连接。

<img src="../nodes/controlaltai/png/ControlAltAI%20Switch.png" width="400" alt="ControlAltAI: Switch (Two Way) / ControlAltAI: Switch (Three Way)">

<a id="controlaltai-perturbation-texture"></a>

### ControlAltAI: Perturbation Texture (扰动纹理)
此节点利用先进的噪声生成技术为图像添加逼真的纹理叠加。该节点特别适用于增强人像、添加胶片颗粒效果或创建自然表面纹理。这是 ControlAltAI: Noise Plus Blend 节点的高级版本。该节点可生成遵循原始图像颜色分布的多通道噪声模式，从而创建能够增强而非掩盖源图像的逼真纹理。可在放大前或放大后使用（像素对像素）。

**参数设置**：<br>
噪声缩放 (noise_scale)：0.25 - 0.50。<br>
纹理强度 (texture_strength)：10 - 50。<br>
扰动因子 (perturbation_factor)：0.10 - 0.25。

节点可在使用或不使用蒙版（mask）的情况下运行。

<img src="../nodes/controlaltai/png/Perturbation%20Texture.png" width="400" alt="ControlAltAI: Perturbation Texture">

**纹理类型**：
- Natural：平衡且有机的纹理——非常适合风格化人像或一般的图像增强，不会掩盖细节。
- Film Grain：添加电影感噪声——非常适合最终渲染或富有创意的电影质感。
- Skin Pore：细致逼真——最适合需要自然面部纹理的特写镜头（close-ups）或人像。
- Fine Details：强调高频纹理——非常适合机械、织物或复杂物体的渲染。

<a id="controlaltai-text-bridge"></a>

### ControlAltAI: Text Bridge (文本桥接)
提供灵活的文本输入/输出管理及手动编辑功能的实用工具节点。此节点作为一个文本处理中心，在接收其他节点文本的同时，也允许进行手动覆写和编辑。

- **透传模式（Passthrough Mode）**：当连接到其他节点且文本输入框为空时，传入的文本将原样输出。
- **手动覆写（Manual Override）**：在文本输入框中输入文本时，将使用输入的内容，而忽略任何透传的输入。
- **独立模式（Standalone Mode）**：在没有进行透传连接时，作为一个简单的文本输入节点运行。

<img src="../nodes/controlaltai/png/Text%20Bridge.png" width="400" alt="ControlAltAI: Text Bridge">

## 许可证

本集成基于 **[gseth/ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes)**（ControlAltAI 节点集合的 Python 3.13 兼容分支）。上游项目采用 **[MIT 许可证](https://github.com/gseth/ControlAltAI-Nodes)** 授权。版权和条款遵循上游仓库和 ControlAltAI 贡献者的规定。
