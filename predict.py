from cv2.gapi import infer
from utils.init_unsloth import FastModel  # should always be the first import: rewuired by unsloth
import os
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from utils.config import Configuration
from utils.dataloaders import test_collate_function,test_collate_function_unsloth
from utils.utilities import visualize_bounding_boxes
from utils.utilities import load_model
from utils.gpu_utils import memory_stats
from utils.logman import logger
import torch

os.makedirs("outputs", exist_ok=True)


def get_dataloader(processor, is_unsloth=False):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    if is_unsloth and FastModel is not None:
        test_collate_fn = partial(
            test_collate_function_unsloth, tokenizer=processor, dtype=cfg.dtype
        )
    else:
        test_collate_fn = partial(
            test_collate_function, processor=processor, dtype=cfg.dtype
        )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn, shuffle=False
    )
    return test_dataloader


if __name__ == "__main__":
    cfg = Configuration.from_args()

    cfg.model_id = cfg.checkpoint_id  # load trained model
    model, tokenizer = load_model(cfg=cfg,isTrain=False)
    if not cfg.use_unsloth:
        processor = AutoProcessor.from_pretrained(cfg.model_id)
    
    model.eval()
    model.to(cfg.device)

    cfg.batch_size = 32
    test_dataloader = get_dataloader(tokenizer if cfg.use_unsloth and FastModel is not None else processor, is_unsloth=True if cfg.use_unsloth and FastModel is not None else False)
    sample, sample_images, image_ids = next(iter(test_dataloader))

    logger.info("before empty cache")
    memory_stats()
    torch.cuda.empty_cache()
    logger.info("after empty cache")
    memory_stats()

    sample = sample.to(model.device)
    generation = model.generate(**sample, max_new_tokens=100)

    logger.info("after inference")
    memory_stats()

    if cfg.use_unsloth and FastModel is not None:
        decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)
    else:
        decoded = processor.batch_decode(generation, skip_special_tokens=True)
    logger.info("After batch decode")
    memory_stats()

    import pickle
    infer_data = {}

    file_count = 0
    for output_text, sample_image, image_id in zip(decoded, sample_images, image_ids):
        image = sample_image[0]
        width, height = image.size
        logger.info(f"image id: {image_id}  model output: {output_text.strip()}")
        try:
            visualize_bounding_boxes(
                image, output_text, width, height, f"outputs/output_{file_count}.png"
            )
            infer_data[file_count] = {
                # "image": image,
                "image_id": image_id,
                "width": width,
                "height": height,
                "output_text": output_text,
            }
            file_count += 1
        except Exception as e:
            logger.info(f"error : {e}")

    with open("outputs/infer_data_unsloth_qlora_300steps.pkl", "wb") as f:
        pickle.dump(infer_data,f)