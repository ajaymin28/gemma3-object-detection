from utils.logman import logger
from utils.init_unsloth import FastModel
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
import torch
from utils.config import Configuration
from utils.gpu_utils import memory_stats
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

def parse_paligemma_label(label, width, height):
    # Extract location codes
    loc_pattern = r"<loc(\d{4})>"
    locations = [int(loc) for loc in re.findall(loc_pattern, label)]

    # Extract category (everything after the last location code)
    category = label.split(">")[-1].strip()

    # Convert normalized locations back to original image coordinates
    # Order in PaliGemma format is: y1, x1, y2, x2
    y1_norm, x1_norm, y2_norm, x2_norm = locations

    # Convert normalized coordinates to actual coordinates
    x1 = (x1_norm / 1024) * width
    y1 = (y1_norm / 1024) * height
    x2 = (x2_norm / 1024) * width
    y2 = (y2_norm / 1024) * height

    return category, [x1, y1, x2, y2]


def visualize_bounding_boxes(image, label, width, height, name):
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Parse the label
    category, bbox = parse_paligemma_label(label, width, height)

    # Draw the bounding box
    draw.rectangle(bbox, outline="red", width=2)

    # Add category label
    draw.text((bbox[0], max(0, bbox[1] - 10)), category, fill="red")

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(draw_image)
    plt.axis("off")
    plt.title(f"Bounding Box: {category}")
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.close()

def str2bool(v):
    """
    Helper function to parse boolean values from cli arguments
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def push_to_hub(model, cfg, tokenizer=None, is_lora=False):
    """
    Push model to huggingface
    """
    push_kwargs = {}
    if tokenizer is not None:
        push_kwargs['tokenizer'] = tokenizer
    model.push_to_hub(cfg.checkpoint_id, **push_kwargs)
    if tokenizer is not None:
        tokenizer.push_to_hub(cfg.checkpoint_id)



def save_best_model(model, cfg, tokenizer=None, is_lora=False):
    """Save LoRA adapter or full model based on config."""
    save_path = f"checkpoints/{cfg.checkpoint_id}_best"
    os.makedirs(save_path, exist_ok=True)
    if is_lora:
        logger.info(f"Saving LoRA adapter to {save_path}")
        model.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
    else:
        logger.info(f"Saving full model weights to {save_path}.pt")
        torch.save(model.state_dict(), f"{save_path}.pt")


def load_saved_model(save_path, cfg, is_lora=False, device=None, logger=None):
    """
    Load LoRA adapter or full model based on config.
    Returns (model, tokenizer)
    """
    tokenizer = None

    if cfg.use_unsloth:

        if logger: logger.info(f"Loading LoRA adapter from {save_path}")

        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name = save_path, # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = True, # Set to False for 16bit LoRA
            device_map="auto"
        )
        return model, tokenizer

    else:
        if is_lora:
            if logger: logger.info(f"Loading LoRA adapter from {save_path}")
            # Load base model first, then LoRA weights
            base_model = AutoModel.from_pretrained(cfg.model_id, device_map=device or "auto")
            model = PeftModel.from_pretrained(base_model, save_path, device_map=device or "auto")
            if os.path.exists(os.path.join(save_path, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(save_path)
        else:
            if logger: logger.info(f"Loading full model weights from {save_path}.pt")
            model = AutoModel.from_pretrained(cfg.model_id, device_map=device or "auto")
            model.load_state_dict(torch.load(f"{save_path}.pt", map_location=device or "cpu"))
            if os.path.exists(os.path.join(save_path, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(save_path)
        return model, tokenizer
    
def load_model(cfg:Configuration, isTrain=True):

    lcfg = cfg.lora
    tokenizer = None

    if cfg.use_unsloth and FastModel is not None:

        # TODO: For LoRA and QLoRa change unsloth config accordigly, generally load_in_4bit, load_in_8bit will be False or LoRA
        model, tokenizer = FastModel.from_pretrained(
            model_name = cfg.model_id,
            max_seq_length = 2048, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = os.environ["HF_TOKEN"] # TODO: Handle this
        )

        if cfg.finetune_method in {"lora", "qlora"} and isTrain:
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = True, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # SHould leave on always!

                r=lcfg.r, # Larger = higher accuracy, but might overfit
                lora_alpha=lcfg.alpha, # Recommended alpha == r at least
                lora_dropout=lcfg.dropout,
                bias = "none",
                random_state = 3407
                # TODO add rs_lora and dora
            )


    else:
        quant_args = {}
        # Enable quantization only for QLoRA
        if cfg.finetune_method in {"qlora"}:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cfg.dtype,
            )
            quant_args = {"quantization_config": bnb_config, "device_map": "auto"}

        from transformers import Gemma3ForConditionalGeneration # import when needed

        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            attn_implementation="eager",
            **quant_args,
        )


        if cfg.finetune_method in {"lora", "qlora"} and isTrain:

            if cfg.finetune_method=="qlora":
                model = prepare_model_for_kbit_training(model)


            lora_cfg = LoraConfig(
                r=lcfg.r,
                lora_alpha=lcfg.alpha,
                target_modules=lcfg.target_modules,
                lora_dropout=lcfg.dropout,
                bias="none",
                use_dora=True if cfg.finetune_method=="qlora" else False,
                use_rslora=True # Rank-Stabilized LoRA  --> `lora_alpha/math.sqrt(r)`
            )
            
            model = get_peft_model(model, lora_cfg)

        elif cfg.finetune_method == "FFT":
            # handled below before printing params
            pass
        else:
            raise ValueError(f"Unknown finetune_method: {cfg.finetune_method}")

    memory_stats()
    torch.cuda.empty_cache() # TODO: Do I need this? Just want to make sure I have mem cleaned up before training starts.
    logger.info(f"called: torch.cuda.empty_cache()")
    memory_stats()
    
    
    for n, p in model.named_parameters():
        if isTrain:
            if cfg.finetune_method == "FFT": # TODO: should FFT finetune all components? or just some, change FFT name to just FT?
                p.requires_grad = any(part in n for part in cfg.mm_tunable_parts)
            if p.requires_grad:
                print(f"{n} will be finetuned")
        else:
            p.requires_grad = False  # laoding for testing
    
    if isTrain:
        if cfg.finetune_method in {"lora", "qlora"}:
            model.print_trainable_parameters()

    return model, tokenizer