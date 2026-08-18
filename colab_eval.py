import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args():
    parser = argparse.ArgumentParser(description="SIDA batch evaluation")
    parser.add_argument("--version", default="saberzl/SIDA-7B")
    parser.add_argument("--image_dir", default="./examples", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args()


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def load_sida_model(
    version="saberzl/SIDA-7B",
    load_in_4bit=True,
    load_in_8bit=False,
    precision="bf16",
    image_size=1024,
    model_max_length=512,
    vision_tower_path="openai/clip-vit-large-patch14",
    conv_type="llava_v1",
    use_mm_start_end=True,
):
    """Load SIDA model once and keep in memory for fast interactive testing."""
    print(">>> Đang nạp mô hình SIDA vào GPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = SIDAForCausalLM.from_pretrained(
        version,
        low_cpu_mem_usage=True,
        vision_tower=vision_tower_path,
        seg_token_idx=seg_token_idx,
        cls_token_idx=cls_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if torch.cuda.is_available():
        model = model.cuda()

    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        v_tower = model.get_model().get_vision_tower()
        v_tower.to(dtype=torch_dtype)
    except AttributeError:
        pass

    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(image_size)
    model.eval()

    print(">>> Nạp mô hình SIDA thành công! Đã sẵn sàng suy luận.")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "clip_image_processor": clip_image_processor,
        "transform": transform,
        "precision": precision,
        "conv_type": conv_type,
        "use_mm_start_end": use_mm_start_end,
    }


def infer_single_image(
    sida_bundle,
    image_path,
    vis_save_path="./vis_output",
    raw_prompt="Please answer begin with [CLS] for classification, if the image is tampered, ouput mask the tampered region.",
):
    """Run fast inference on a single image without reloading the model."""
    os.makedirs(vis_save_path, exist_ok=True)
    model = sida_bundle["model"]
    tokenizer = sida_bundle["tokenizer"]
    clip_image_processor = sida_bundle["clip_image_processor"]
    transform = sida_bundle["transform"]
    precision = sida_bundle["precision"]
    conv_type = sida_bundle["conv_type"]
    use_mm_start_end = sida_bundle["use_mm_start_end"]

    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + raw_prompt
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"Lỗi: Không tìm thấy hoặc không đọc được ảnh tại {image_path}")
        return None

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if precision == "bf16":
        image = image.bfloat16()
    elif precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    print("text_output: ", text_output)

    base_name = os.path.basename(image_path).split(".")[0]
    if "tampered" in text_output.lower():
        cls_label = "tampered"
    elif "real" in text_output.lower():
        cls_label = "real"
    else:
        cls_label = "synthetic"

    saved_files = []
    if len(pred_masks) > 0:
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = f"{vis_save_path}/{base_name}_mask_{i}.jpg"
            cv2.imwrite(save_path, pred_mask * 255)
            saved_files.append(save_path)
            print(f"{save_path} has been saved.")

            save_path = f"{vis_save_path}/{base_name}_masked_img_{i}.jpg"
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            saved_files.append(save_path)
            print(f"{save_path} has been saved.")
    else:
        h, w = image_np.shape[:2]
        blank_mask = np.zeros((h, w), dtype=np.uint8)

        save_path = f"{vis_save_path}/{base_name}_mask_0.jpg"
        cv2.imwrite(save_path, blank_mask)
        saved_files.append(save_path)

        save_path = f"{vis_save_path}/{base_name}_masked_img_0.jpg"
        save_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        saved_files.append(save_path)
        print(f"[{cls_label}] {base_name} — no mask, saved original.")

    return {
        "text_output": text_output,
        "classification": cls_label,
        "saved_files": saved_files,
    }


def main():
    args = parse_args()
    os.makedirs(args.vis_save_path, exist_ok=True)

    valid_exts = [".jpg", ".jpeg", ".png", ".webp"]
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext.upper()}")))

    print(f"Tìm thấy {len(image_paths)} ảnh trong {args.image_dir}")
    if len(image_paths) == 0:
        return

    sida_bundle = load_sida_model(
        version=args.version,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        precision=args.precision,
        image_size=args.image_size,
        model_max_length=args.model_max_length,
        vision_tower_path=args.vision_tower,
        conv_type=args.conv_type,
        use_mm_start_end=args.use_mm_start_end,
    )

    for image_path in image_paths:
        print(f"\nĐang xử lý: {image_path}")
        infer_single_image(
            sida_bundle, image_path, vis_save_path=args.vis_save_path
        )


if __name__ == "__main__":
    main()
