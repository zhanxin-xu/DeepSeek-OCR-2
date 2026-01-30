# DeepSeek-OCR-2 批量处理性能测试指南

本指南介绍如何使用 `batch_ocr_pdf.py` 脚本对 PDF 文件进行批量 OCR 处理和性能测试。

## 1. 准备工作

### 硬件要求
*   **GPU**: 推荐 RTX 30/40 系列 (Ampere/Ada 架构)，以支持 Flash Attention 2 和 BF16 加速。
    *   *当前配置*: 已适配 Flash Attention 2 + BFloat16，直接 GPU 加载。
*   **显存**: 建议 12GB 或以上。

### 软件环境
*   确保已安装以下关键库：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install transformers accelerate flash_attn
    pip install pymupdf
    ```
    *   *注意*: `flash_attn` 通常需要预编译的 wheel 包。

### 数据准备
1.  将需要测试的 PDF 文件放入根目录下的 `data` 文件夹中。
    *   例如：`E:\code\DeepSeek-OCR-2\data\sample.pdf`

## 2. 运行脚本

### 2.1 快速性能测试 (推荐)
为了快速验证模型速度和显存占用，可以使用 `--max_pages` 参数只处理前几页。

```powershell
python batch_ocr_pdf.py --max_pages 5
```
*   **作用**: 对 data 目录下的每个 PDF 只处理前 5 页。
*   **用途**: 下班前确认环境正常，获取单页平均耗时 (Average Time per Page)。

### 2.2 完整处理
当需要转换整个文档时，不加任何参数运行：

```powershell
python batch_ocr_pdf.py
```

## 3. 输出结果

所有结果将保存在 `output` 目录下，按 PDF 文件名分类。

目录结构示例：
```text
output/
└── my_document_name/
    ├── ocr_stats.json          <-- 性能统计报告 (含总耗时、平均耗时)
    └── results/
        ├── page_1/
        │   ├── source_image.jpg         <-- 原始页面截图 (2x放大)
        │   ├── content.mmd              <-- 识别结果 (Markdown格式)
        │   └── visualization.jpg        <-- 可视化结果 (带检测框)
        ├── page_2/
        │   ...
```

### 3.1 查看性能统计
打开 `output/[pdf_name]/ocr_stats.json` 查看详细性能数据：
*   **total_time_seconds**: 总处理时间
*   **average_time_per_page**: 平均单页推理时间 (衡量 GPU 性能的关键指标)
*   **pages**: 每一页的详细处理状态和耗时

## 4. 常见问题 (FAQ)

### Q: 为什么第一页处理很慢？
**A**: 第一页通过模型时会触发 PyTorch 的 JIT (Just-In-Time) 编译和内核初始化。这是正常现象，从第二页开始速度会稳定下来。

### Q: 如何指定特定的 GPU？
**A**: 脚本默认使用系统可见的第一个 GPU (cuda:0)。如果你有多个显卡，可以在运行前设置环境变量：
```powershell
$env:CUDA_VISIBLE_DEVICES="1"; python batch_ocr_pdf.py
```
*(注: PowerShell设置方式)*
