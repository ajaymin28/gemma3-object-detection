import numpy as np
from utils.logman import logger
from utils.config import Configuration
from utils.create_dataset import format_objects

def train_collate_function_unsloth(batch_of_samples, tokenizer, dtype, transform=None):
    """
    unsloth
    """
    images = []
    prompts = []
    for sample in batch_of_samples:
        if transform:
            transformed = transform(
                image=np.array(sample["image"]),
                bboxes=sample["objects"]["bbox"],
                category_ids=sample["objects"]["category"]
            )
            sample["image"] = transformed["image"]
            sample["objects"]["bbox"] = transformed["bboxes"]
            sample["objects"]["category"] = transformed["category_ids"]
            sample["height"] = sample["image"].shape[0]
            sample["width"] = sample["image"].shape[1]
            sample['label_for_paligemma'] = format_objects(sample)['label_for_paligemma'] 
        images.append([sample["image"]])
        prompts.append(
            f"{tokenizer.boi_token} detect \n\n{sample['label_for_paligemma']} {tokenizer.eos_token}"
        )

    # Use tokenizer directly (Unsloth tokenizer supports vision inputs for Gemma3)
    batch = tokenizer(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True
    )

    labels = batch["input_ids"].clone()

    # Mask out padding, image tokens, and other special tokens from loss
    image_token_id = [
        tokenizer.tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
    ]
    labels[labels == tokenizer.pad_token_id] = -100
    for tok_id in image_token_id:
        labels[labels == tok_id] = -100
    labels[labels == 262144] = -100  # If this ID is used for your "unused" special token

    batch["labels"] = labels
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(dtype)

    return batch

def train_collate_function(batch_of_samples, processor, dtype, transform=None):
    images = []
    prompts = []
    for sample in batch_of_samples:
        if transform:
            transformed = transform(image=np.array(sample["image"]), bboxes=sample["objects"]["bbox"], category_ids=sample["objects"]["category"])
            sample["image"] = transformed["image"]
            sample["objects"]["bbox"] = transformed["bboxes"]
            sample["objects"]["category"] = transformed["category_ids"]
            sample["height"] = sample["image"].shape[0]
            sample["width"] = sample["image"].shape[1]
            sample['label_for_paligemma'] = format_objects(sample)['label_for_paligemma'] 
        images.append([sample["image"]])
        prompts.append(
            f"{processor.tokenizer.boi_token} detect \n\n{sample['label_for_paligemma']} {processor.tokenizer.eos_token}"
        )

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels

    # List from https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels

    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    return batch

def test_collate_function_unsloth(batch_of_samples, tokenizer, dtype):
    images = []
    prompts = []
    image_ids = []
    for sample in batch_of_samples:
        images.append([sample["image"]])
        prompts.append(f"{tokenizer.tokenizer.boi_token} detect \n\n")
        image_ids.append(sample["image_id"])

    # Use tokenizer directly (Unsloth tokenizer supports vision inputs for Gemma3)
    batch = tokenizer(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True
    )

    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    return batch, images, image_ids

def test_collate_function(batch_of_samples, processor, dtype):
    images = []
    prompts = []
    image_ids = []
    for sample in batch_of_samples:
        images.append([sample["image"]])
        prompts.append(f"{processor.tokenizer.boi_token} detect \n\n")
        image_ids.append(sample["image_id"])

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    batch["pixel_values"] = batch["pixel_values"].to(
        dtype
    )  # to check with the implementation
    return batch, images, image_ids

def get_dataloader(cfg:Configuration,processor=None,tokenizer=None, split="train", is_unsloth=False):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from functools import partial
    from utils.data_aug import augmentations

    logger.info(f"Fetching the dataset: {cfg.dataset_id}:{split}")
    train_dataset = load_dataset(cfg.dataset_id, split=split)

    if is_unsloth:
        # <- Use the Unsloth tokenizer instead of processor
        train_collate_fn = partial(train_collate_function_unsloth,tokenizer=tokenizer,dtype=cfg.dtype,transform=augmentations)
    else:
        train_collate_fn = partial(train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations)

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    return train_dataloader