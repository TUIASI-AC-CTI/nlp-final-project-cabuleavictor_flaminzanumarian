import numpy as np
from typing import Union, List, Dict
import torch


def predict(
        text: Union[str, List[str]],
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        threshold: float = 0.4,
        return_attention: bool = False,
        temperature: float = 1.0,
        return_logits: bool = False,
        calibration_factor: float = 1.2
) -> Union[Dict, List[Dict]]:
    model.eval()

    if not isinstance(text, (str, list, tuple)):
        raise ValueError("Input must be a string or list of strings")

    is_batch = isinstance(text, (list, tuple))
    texts = [text] if not is_batch else text

    if len(texts) == 0:
        return [] if is_batch else {}

    try:
        inputs = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding='longest',
            max_length=256,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True
        ).to(device)
    except Exception as e:
        raise ValueError(f"Tokenization failed: {str(e)}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(**inputs, output_attentions=return_attention)

        scaled_logits = outputs.logits / temperature
        if calibration_factor != 1.0:
            scaled_logits[:, 1] *= calibration_factor

        probs = torch.softmax(scaled_logits, dim=-1)
        hate_probs = probs[:, 1].cpu().numpy()
        hate_probs = np.clip(hate_probs, 1e-5, 1 - 1e-5)

        if return_attention:
            attentions = torch.stack(outputs.attentions).mean(dim=0).mean(dim=1).cpu().numpy()

    results = []
    for i, (text, hate_prob) in enumerate(zip(texts, hate_probs)):
        prediction = 'Hate' if hate_prob > threshold else 'Not Hate'
        confidence = hate_prob if prediction == 'Hate' else 1 - hate_prob
        confidence = np.clip(confidence, 0.01, 0.99)

        log_odds = np.log(hate_prob / (1 - hate_prob + 1e-10))

        result = {
            'text': text,
            'prediction': prediction,
            'confidence': float(confidence),
            'hate_probability': float(hate_prob),
            'not_hate_probability': float(1 - hate_prob),
            'log_odds': float(log_odds),
            'threshold_used': float(threshold)
        }

        if return_logits:
            result['logits'] = scaled_logits[i].cpu().numpy().tolist()

        if return_attention:
            input_ids = inputs['input_ids'][i]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            special_ids = set(tokenizer.all_special_ids)
            valid_indices = [j for j, token_id in enumerate(input_ids) if token_id.item() not in special_ids]
            result['attention_weights'] = {
                'tokens': [tokens[j] for j in valid_indices],
                'weights': [attentions[i][j] for j in valid_indices]
            }

        results.append(result)

    return results[0] if not is_batch else results
