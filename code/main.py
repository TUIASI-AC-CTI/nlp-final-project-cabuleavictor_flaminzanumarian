import pandas as pd
import torch
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
import logging
from torch.optim import AdamW
from dataset_preprocessing import HateSpeechDataset, load_and_combine_datasets
from model import ModelTrainer
from generate_prediction import predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hate_speech_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_data_loaders(train_df, val_df, test_df, tokenizer, batch_size=32):
    train_dataset = HateSpeechDataset(train_df['text'], train_df['label'], tokenizer)
    val_dataset = HateSpeechDataset(val_df['text'], val_df['label'], tokenizer)
    test_dataset = HateSpeechDataset(test_df['text'], test_df['label'], tokenizer)

    class_counts = np.bincount(train_df['label'])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_df['label'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, val_loader, test_loader

def initialize_model(model_name, num_labels=2, frozen_layers=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True
    )

    if frozen_layers is not None:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.bert.embeddings.parameters():
            param.requires_grad = True

        for layer_num in frozen_layers:
            for param in model.bert.encoder.layer[layer_num].parameters():
                param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    return model

def evaluate(model, test_loader, device):
    model.eval()
    preds, truths, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=1)
            preds.extend(torch.argmax(batch_probs, dim=1).cpu().numpy())
            truths.extend(batch['labels'].cpu().numpy())
            probs.extend(batch_probs[:, 1].cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(truths, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info("\nTest Evaluation Metrics:")
    logger.info(classification_report(truths, preds, target_names=['Not Hate', 'Hate']))
    logger.info(f"\nOptimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Max F1 at threshold: {f1_scores[optimal_idx]:.4f}")

    return optimal_threshold

def main():
    set_seed()

    logger.info("Loading and preparing datasets...")
    #df = load_and_combine_datasets()
    df = pd.read_csv('processed_data_final.csv')

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df['label'],
        random_state=42
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df['label'],
        random_state=42
    )

    logger.info("\nDataset sizes:")
    logger.info(f"Train: {len(train_df)}")
    logger.info(f"Validation: {len(val_df)}")
    logger.info(f"Test: {len(test_df)}")

    model_name = "GroNLP/hateBERT"
    logger.info(f"Initializing model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer max length: {tokenizer.model_max_length}")

    initial_frozen_layers = list(range(11))
    model = initialize_model(model_name, frozen_layers=initial_frozen_layers)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda:2':
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(device.index)}")
    model = model.to(device)

    batch_size = 32
    gradient_accumulation_steps = 2
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, batch_size
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=1e-4,
        eps=1e-8,
    )

    n_epochs = 20
    total_steps = len(train_loader) * n_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    trainer_config = {
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'n_epochs': n_epochs,
        'patience': 3,
        'max_grad_norm': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'unfreeze_schedule': {
            'epochs': [2, 4, 6, 8, 10],
            'layers': [
                [10, 11],
                [8, 9, 10, 11],
                [6, 7, 8, 9, 10, 11],
                [4, 5, 6, 7, 8, 9, 10, 11],
                list(range(12))
            ]
        }
    }

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        config=trainer_config
    )

    logger.info("\nStarting training...")
    logger.info("Initial trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"- {name}")

    best_metrics = trainer.train(train_loader, val_loader, test_loader)
    logger.info(f"\nOptimal Threshold: {best_metrics}")

    threshold = 0.5

    test_samples = pd.read_csv("additional_text_cases.csv")

    logger.info("\nTesting on sample texts:")
    for text in test_samples:
        result = predict(text, model, tokenizer, device, threshold)
        logger.info(f"\nText: {result['text']}")
        logger.info(f"Prediction: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        logger.info(f"Hate probability: {result['hate_probability']:.4f}")

    logger.info("\nTraining complete.")

if __name__ == "__main__":
    main()
