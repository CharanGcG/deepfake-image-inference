# engine/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from code.utils.metrics import compute_metrics
from code.utils.checkpoint import save_checkpoint
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: str, scaler: GradScaler, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Starting training epoch...")
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % 500 == 0:
            logger.info(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        images, labels, _ = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, auc, precision, recall, f1 = compute_metrics(all_labels, all_preds, all_probs)
    logger.info(f"Training epoch completed. Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: str, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Starting evaluation...")
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if (batch_idx + 1) % 500 == 0:
                logger.info(f"Evaluating batch {batch_idx+1}/{len(dataloader)}")
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, auc, precision, recall, f1 = compute_metrics(all_labels, all_preds, all_probs)
    logger.info(f"Evaluation completed. Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_phase(phase_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              criterion, optimizer, scheduler, device: str, num_epochs: int,
              run_dir: str, best_auc: float, scaler: GradScaler, logger: logging.Logger,
              patience: int = 5) -> float:
    logger.info(f"Running training phase: {phase_name}")

    epochs_no_improve = 0  # Track epochs without improvement

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} started")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, logger)
        val_metrics = evaluate(model, val_loader, criterion, device, logger)

        if scheduler is not None:
            try:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("loss", 0.0))
                else:
                    scheduler.step()
            except Exception as e:
                logger.warning(f"Scheduler step failed: {e}")

        logger.info(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} completed")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Validation Metrics: {val_metrics}")

        # Clear GPU cache after each epoch
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after epoch")

        # Create checkpoint state
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_metrics": val_metrics,
            "best_auc": best_auc,
        }

        # Save last checkpoint
        save_checkpoint(state, is_best=False, output_dir=run_dir)
        logger.info("Checkpoint saved (last)")

        # Save best if improved and reset early stopping counter
        current_auc = val_metrics.get("auc", 0.0)
        if current_auc > best_auc:
            best_auc = current_auc
            save_checkpoint(state, is_best=True, output_dir=run_dir)
            logger.info(f"New best model saved with AUC {best_auc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in AUC for {epochs_no_improve} epoch(s)")

        # Early stopping check
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs without improvement")
            break

    logger.info(f"Phase {phase_name} completed with best AUC {best_auc:.4f}")
    return best_auc
