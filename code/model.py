import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    accuracy_score,
)
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss


class ModelTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR, device: torch.device,
                 scaler: torch.cuda.amp.GradScaler, config: Dict):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.config = config
        self.current_epoch = 0

        self.criterion = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            reduction='mean'
        )

        self.best_metrics = {
            'f1': 0,
            'pr_auc': 0,
            'roc_auc': 0,
            'epoch': 0
        }
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_f1_history = []
        self.val_metrics_history = []
        self.unfreezing_history = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"model_output_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        self._log_trainable_parameters()

    def _log_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params / total_params:.1%})")

    def _plot_training_metrics(self):
        epochs = range(1, len(self.train_loss_history) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label='Train Loss')
        if len(self.val_loss_history) == len(self.train_loss_history):
            plt.plot(epochs, self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if len(self.val_f1_history) == len(self.train_loss_history):
            plt.plot(epochs, self.val_f1_history, label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training metrics plot saved to {plot_path}")

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for step, batch in enumerate(progress_bar):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels) / self.config['gradient_accumulation_steps']

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            progress_bar.set_postfix({'loss': loss.item() * self.config['gradient_accumulation_steps']})

        avg_loss = total_loss / len(train_loader)
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def evaluate(self, data_loader: torch.utils.data.DataLoader, mode: str = 'val') -> Dict:
        self.model.eval()
        preds, truths, probs = [], [], []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating ({mode})", leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs.logits, labels)

                total_loss += loss.item()
                batch_probs = torch.softmax(outputs.logits, dim=1)
                preds.extend(torch.argmax(batch_probs, dim=1).cpu().numpy())
                truths.extend(batch['labels'].cpu().numpy())
                probs.extend(batch_probs[:, 1].cpu().numpy())

        metrics = self._compute_metrics(preds, truths, probs, mode)
        avg_loss = total_loss / len(data_loader)

        if mode == 'val':
            self.val_loss_history.append(avg_loss)
            self.val_f1_history.append(metrics['f1'])

        return metrics

    def _compute_metrics(self, preds: List, truths: List, probs: List, mode: str = 'val') -> Dict:
        metrics = {
            'accuracy': accuracy_score(truths, preds),
            'f1': f1_score(truths, preds, average='macro'),
            'roc_auc': roc_auc_score(truths, probs),
        }

        precision, recall, thresholds = precision_recall_curve(truths, probs)
        metrics['pr_auc'] = auc(recall, precision)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        metrics['optimal_threshold'] = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        metrics['optimal_f1'] = f1_scores[optimal_idx]

        logger.info(f"\n{mode.capitalize()} Classification Report:")
        logger.info(classification_report(truths, preds, target_names=['Not Hate', 'Hate']))

        self._plot_confusion_matrix(truths, preds, mode)

        if mode == 'val':
            self.val_metrics_history.append(metrics)

        return metrics

    def _plot_confusion_matrix(self, truths: List, preds: List, mode: str):
        cm = confusion_matrix(truths, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Hate', 'Hate'],
                    yticklabels=['Not Hate', 'Hate'])
        plt.title(f'{mode.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plot_path = os.path.join(self.output_dir, f'{mode}_confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {plot_path}")

    def save_model(self, epoch: int, metrics: Dict):
        model_path = os.path.join(self.output_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'focal_alpha': self.criterion.alpha,
            'focal_gamma': self.criterion.gamma,
            'unfreezing_history': self.unfreezing_history,
        }, model_path)
        logger.info(f"Model checkpoint saved to {model_path}")

    def _unfreeze_layers(self, layer_indices: List[int]):
        for param in self.model.bert.encoder.layer.parameters():
            param.requires_grad = False

        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = True

        for layer_num in layer_indices:
            for param in self.model.bert.encoder.layer[layer_num].parameters():
                param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        logger.info(f"Unfrozen layers: {layer_indices} (plus embeddings)")
        self.unfreezing_history.append({
            'epoch': self.current_epoch,
            'layers': layer_indices
        })
        self._log_trainable_parameters()

    def _check_unfreeze_schedule(self):
        if 'unfreeze_schedule' not in self.config:
            return

        schedule = self.config['unfreeze_schedule']
        for epoch, layers in zip(schedule['epochs'], schedule['layers']):
            if self.current_epoch == epoch:
                self._unfreeze_layers(layers)
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.optimizer.param_groups[0]['lr'],
                    weight_decay=self.config.get('weight_decay', 1e-4),
                    eps=self.config.get('eps', 1e-8),
                )
                logger.info("Optimizer reinitialized with new trainable parameters")
                break

    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              test_loader: Optional[torch.utils.data.DataLoader] = None) -> dict:

        for epoch in range(self.config['n_epochs']):
            self.current_epoch = epoch + 1
            logger.info(f"\n{'=' * 30} Epoch {self.current_epoch}/{self.config['n_epochs']} {'=' * 30}")

            self._check_unfreeze_schedule()

            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.evaluate(val_loader, 'val')
            logger.info(f"Val Metrics - F1: {val_metrics['f1']:.4f}, "
                        f"PR AUC: {val_metrics['pr_auc']:.4f}, "
                        f"ROC AUC: {val_metrics['roc_auc']:.4f}")

            if val_metrics['f1'] > self.best_metrics['f1']:
                self.best_metrics = {
                    **val_metrics,
                    'epoch': self.current_epoch
                }
                self.save_model(self.current_epoch, val_metrics)
                logger.info("New best model saved!")

            self._plot_training_metrics()

            if (self.current_epoch - self.best_metrics['epoch']) >= self.config['patience']:
                logger.info(f"Early stopping at epoch {self.current_epoch}")
                break

        if test_loader:
            logger.info("\nRunning final evaluation on test set...")
            test_metrics = self.evaluate(test_loader, 'test')
            logger.info(f"Test Metrics - F1: {test_metrics['f1']:.4f}, "
                        f"PR AUC: {test_metrics['pr_auc']:.4f}, "
                        f"ROC AUC: {test_metrics['roc_auc']:.4f}")

            return test_metrics

        return self.best_metrics
