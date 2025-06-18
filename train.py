from utils.logman import logger
import wandb
import torch
from torch.amp import autocast, GradScaler
from transformers import AutoProcessor

from utils.config import Configuration
from utils.utilities import load_model
from utils.utilities import save_best_model, push_to_hub, load_saved_model
from utils.gpu_utils import memory_stats
from utils.utilities import get_dataloader

def step(model, batch, device, use_fp16, optimizer=None, scaler=None):
    """
    Single batch process
    """
    data = batch.to(device)
    if use_fp16:
        with autocast(device_type=device):
            loss = model(**data).loss
    else:
        loss = model(**data).loss
    if optimizer:
        optimizer.zero_grad()
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    return loss.item()

def validate_all(model, val_loader, cfg, use_fp16,val_batches=5):

    # if cfg.use_unsloth and FastModel is not None:
    #     FastModel.for_inference(model) # Enable for inference!
    # else:
    #     model.eval()
    model.eval()

    with torch.no_grad():
        if val_batches>0:
            ## TODO: This logic is Temp and should be removed in final clean up
            n_batches = val_batches
            losses = []
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                losses.append(step(model, batch, cfg.device, use_fp16))
        else:
            losses = [step(model, batch, cfg.device, use_fp16) for batch in val_loader]
    model.train()
    return sum(losses) / len(losses) if len(losses)> 0 else 0


def train_model(model, optimizer, cfg:Configuration, train_loader, val_loader=None):

    memory_stats()
    torch.cuda.empty_cache() # TODO: Do I need this? Just want to make sure I have mem cleaned up before training starts.
    logger.info(f"called: torch.cuda.empty_cache()")
    memory_stats()
    use_fp16 = cfg.dtype in [torch.float16, torch.bfloat16]
    scaler = GradScaler() if use_fp16 else None
    global_step, best_val_loss = 0, float("inf")


    # if cfg.use_unsloth and FastModel is not None:
    #     # logger.info("Before setting for training...")
    #     # model.print_trainable_parameters()
    #     FastModel.for_training(model, use_gradient_checkpointing=True) # Enable for training!  # TODO :calling this method uses so much memory, investigate
    # else:
    #     model.train()
    #     model.to(cfg.device)

    model.train()
    model.to(cfg.device)

    logger.info("after setting for training...")
    # model.print_trainable_parameters()


    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_loader):

            torch.cuda.reset_peak_memory_stats(cfg.device) 
            torch.cuda.empty_cache()

            loss = step(model, batch, cfg.device, use_fp16, optimizer, scaler)
            if global_step % 1 == 0:
                logger.info(f"Epoch:{epoch} Step:{global_step} Loss:{loss:.4f}")
                wandb.log({"train/loss": loss, "epoch": epoch}, step=global_step)
            if val_loader and global_step % cfg.validate_steps_freq == 0:
                val_loss = validate_all(model, val_loader, cfg, use_fp16, val_batches=1) # if val_batches>0 the code will validate on that many batches only. -1 to disable this
                logger.info(f"Step:{global_step} Val Loss:{val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"})

            ## Model seem to converge before even first epoch finishes for LoRA. set max_step_to_train<=0 to disable this.
            if global_step>cfg.max_step_to_train-1 and cfg.max_step_to_train>0:
                break
            global_step += 1
            if global_step % 5 == 0:
                memory_stats()
            
            if global_step % 5 == 0:
                memory_stats()

    return model





if __name__ == "__main__":
    # 1. Parse CLI + YAMLs into config
    cfg = Configuration.from_args()  # config.yaml is overriden by CLI arguments

    # 2. Load model
    logutils.info(f"Getting model for {cfg.finetune_method}")
    # loads model based on config. Unsloth, lora, qlora, FFT
    model, tokenizer = load_model(cfg)

    # 3. Get Data
    if cfg.use_unsloth:
        train_dataloader = get_dataloader(args=cfg,tokenizer=tokenizer, split="train",is_unsloth=True)
        validation_dataloader = get_dataloader(args=cfg,tokenizer=tokenizer, split="validation",is_unsloth=True)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model_id)
        train_dataloader = get_dataloader(args=cfg, processor=processor, split="train")
        validation_dataloader = get_dataloader(args=cfg, processor=processor, split="validation")
    
    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    # 5. Enable logging, need to login or set wanddb token in os.env
    wandb.init(
        project=cfg.wandb_project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else None,
        config=vars(cfg),
    )

    # 5. Actual train and validation, validation_dataloader=None to do just traing.
    train_model(model, optimizer, cfg, train_dataloader, validation_dataloader)

    # 6. Loading best model back
    model, tokenizer = load_saved_model(cfg, is_lora=cfg.finetune_method in {"lora", "qlora"}, device="cuda", logger=logutils)
    logutils.info(f"Pushing to hub at: {cfg.checkpoint_id}")
    if cfg.push_model_to_hub:
        push_to_hub(model, cfg, tokenizer, cfg.finetune_method in {"lora", "qlora"})

    # 7. Test?  # TODO
    
    
    # 8. Wrap up
    wandb.finish()
    logutils.info("Train finished")
