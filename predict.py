import os
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from utils.init_unsloth import FastModel
from utils.config import Configuration
from utils.dataloaders import test_collate_function,test_collate_function_unsloth
from utils.utilities import visualize_bounding_boxes
from train import load_model

os.makedirs("outputs", exist_ok=True)


def get_dataloader(processor, is_unsloth=False):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    if is_unsloth and FastModel is not None:
        test_collate_fn = partial(
            test_collate_function_unsloth, processor=processor, dtype=cfg.dtype
        )
    else:
        test_collate_fn = partial(
            test_collate_function, processor=processor, dtype=cfg.dtype
        )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


if __name__ == "__main__":
    cfg = Configuration.from_args()

    
    model, tokenizer = load_model(cfg=cfg,isTrain=False)
    if cfg.use_unsloth and FastModel is not None:
        processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    # model = Gemma3ForConditionalGeneration.from_pretrained(
    #     cfg.checkpoint_id,
    #     torch_dtype=cfg.dtype,
    #     device_map="cpu",
    # )
    model.eval()
    model.to(cfg.device)


    test_dataloader = get_dataloader(processor=tokenizer if cfg.use_unsloth and FastModel is not None else processor)
    sample, sample_images = next(iter(test_dataloader))
    sample = sample.to(cfg.device)

    generation = model.generate(**sample, max_new_tokens=100)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)

    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        width, height = image.size
        visualize_bounding_boxes(
            image, output_text, width, height, f"outputs/output_{file_count}.png"
        )
        file_count += 1
