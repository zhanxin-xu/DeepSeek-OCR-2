
import os
import glob
import time
import json
import argparse
import torch
import fitz  # PyMuPDF
from transformers import AutoModel, AutoTokenizer

# Configuration
# Configuration
DATA_DIR = os.path.abspath('data')
OUTPUT_DIR = os.path.abspath('output')
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'

def setup_model():
    print("Loading DeepSeek-OCR-2 model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # 使用 Flash Attention 2 加速推理
        # 优化：直接在 GPU 上以 bfloat16 加载，避免 CPU->GPU 的转换开销和警告
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            attn_implementation='flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model = model.eval()
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

def process_pdf(pdf_path, tokenizer, model, max_pages=None):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Root output directory for this PDF
    pdf_root_dir = os.path.join(OUTPUT_DIR, pdf_name)
    
    # Subdirectories for structure
    results_dir = os.path.join(pdf_root_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nProcessing PDF: {pdf_name}")
    print(f"Output directory: {pdf_root_dir}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return

    total_pages = len(doc)
    stats = {
        "pdf_name": pdf_name,
        "total_pages": total_pages,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pages": [],
        "total_time_seconds": 0,
        "average_time_per_page": 0
    }
    
    print(f"Total pages: {total_pages}")
    if max_pages:
        print(f"Test mode active: processing first {max_pages} pages only.")
    
    pdf_start_time = time.time()
    
    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        page_num = i + 1
        
        # Create a dedicated directory for this page
        page_dir = os.path.join(results_dir, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)
        
        page_stats = {"page": page_num}
        
        try:
            # 1. Convert PDF page to Image
            # Using Matrix(2, 2) for 2x zoom (approx 144 dpi) for better OCR accuracy
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image_filename = f"source_image.jpg"
            image_path = os.path.join(page_dir, image_filename)
            pix.save(image_path)
            
            page_stats["image_path"] = os.path.join("results", f"page_{page_num}", image_filename)
            
            # 2. Perform OCR
            print(f"  [Page {page_num}/{total_pages}] Running OCR...", end="", flush=True)
            ocr_start_time = time.time()
            
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            # Run inference
            # We use page_dir as output_path so model saves intermediate files there
            res = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_path, 
                output_path=page_dir, 
                base_size=1024, 
                image_size=768, 
                crop_mode=True, 
                save_results=True
            )
            
            ocr_end_time = time.time()
            duration = ocr_end_time - ocr_start_time
            print(f" Done ({duration:.2f}s)")
            
            page_stats["duration_seconds"] = duration
            page_stats["status"] = "success"
            
            # 3. Handle Output Files
            # Model saves 'result.mmd' and 'result_with_boxes.jpg' in output_path (page_dir)
            default_mmd = os.path.join(page_dir, "result.mmd")
            default_vis = os.path.join(page_dir, "result_with_boxes.jpg")
            
            target_mmd = os.path.join(page_dir, "content.mmd")
            target_vis = os.path.join(page_dir, "visualization.jpg")
            
            if os.path.exists(default_mmd):
                if os.path.exists(target_mmd):
                    os.remove(target_mmd)
                os.rename(default_mmd, target_mmd)
                page_stats["result_file"] = os.path.join("results", f"page_{page_num}", "content.mmd")
            else:
                page_stats["result_file"] = "not_found"
                
            if os.path.exists(default_vis):
                if os.path.exists(target_vis):
                    os.remove(target_vis)
                os.rename(default_vis, target_vis)
                page_stats["vis_file"] = os.path.join("results", f"page_{page_num}", "visualization.jpg")

            
        except Exception as e:
            print(f" Failed! Error: {e}")
            page_stats["status"] = "failed"
            page_stats["error"] = str(e)
        
        stats["pages"].append(page_stats)

    pdf_end_time = time.time()
    total_time = pdf_end_time - pdf_start_time
    stats["total_time_seconds"] = total_time
    stats["average_time_per_page"] = total_time / total_pages if total_pages > 0 else 0
    
    # Save statistics at the root of the PDF output directory
    stats_path = os.path.join(pdf_root_dir, "ocr_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Finished {pdf_name}. Total: {total_time:.2f}s, Avg: {stats['average_time_per_page']:.2f}s/page")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Batch OCR PDF processing")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to process per PDF. If not set, process all pages.")
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' does not exist.")
        return
        
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    # Initialize model once
    tokenizer, model = setup_model()
    
    for pdf_file in pdf_files:
        process_process(pdf_file, tokenizer, model, args.max_pages)

def process_process(pdf_file, tokenizer, model, max_pages=None):
     process_pdf(pdf_file, tokenizer, model, max_pages)

if __name__ == "__main__":
    main()
```
