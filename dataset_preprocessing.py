import torch
from datasets import load_dataset
import re
import unicodedata
import pandas as pd
from torch.utils.data import Dataset
import contractions

import nltk
from nltk.corpus import wordnet
import random
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import words
nltk.download('words')

english_vocab = set(words.words())

def synonym_augment(text, num_replacements=1):
    words = text.split()
    new_words = words[:]
    random.shuffle(new_words)
    replaced = 0

    for i, word in enumerate(new_words):
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_words = [lemma.name().replace('_', ' ') for lemma in synonyms[0].lemmas()
                             if lemma.name().lower() != word.lower()]
            if synonym_words:
                replacement = random.choice(synonym_words)
                index = words.index(word)
                words[index] = replacement
                replaced += 1
        if replaced >= num_replacements:
            break
    return ' '.join(words)

def reduce_repeated_letters(word):
    if word in english_vocab:
        return word

    reduced_word = re.sub(r'(.)\1{2,}', r'\1\1', word)

    if reduced_word in english_vocab:
        return reduced_word
    else:
        return word

def strip_leading_punctuation(text):
    return re.sub(r'^[^\w]+', '', text)


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        lengths = [len(tokenizer(text, truncation=False)['input_ids']) for text in texts]
        print(max(lengths), sum(lengths) / len(lengths))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts.iloc[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


def load_and_combine_datasets():
    datasets = []

    try:
        print("Trying to load HateXplain...")
        hatexplain = load_dataset("hatexplain")
        texts = [' '.join(tokens) for tokens in hatexplain['train']['post_tokens']]
        labels = [
            1 if max(set(labs['label']), key=labs['label'].count) > 0 else 0
            for labs in hatexplain['train']['annotators']
        ]
        df_hate = pd.DataFrame({'text': texts, 'label': labels})
        datasets.append(df_hate)
        print(f"Loaded {len(df_hate)} HateXplain samples")
    except Exception as e:
        print(f"Could not load HateXplain: {str(e)}")

    try:
        print("Trying to load Davidson...")
        davidson = load_dataset("ucberkeley-dlab/measuring-hate-speech", 'default')
        df_davidson = pd.DataFrame(davidson['train'])
        df_davidson['label'] = df_davidson['hate_speech_score'].apply(lambda x: 1 if x > 0.5 else 0)
        df_davidson = df_davidson[['text', 'label']]
        datasets.append(df_davidson)
        print(f"Loaded {len(df_davidson)} Davidson samples")
    except Exception as e:
        print(f"Could not load Davidson: {str(e)}")

    try:
        print("Loading OLID dataset from christophsonntag/OLID...")
        olid = load_dataset("christophsonntag/OLID")
        df_olid = pd.concat([pd.DataFrame(olid['train']), pd.DataFrame(olid['test'])], ignore_index=True)
        df_olid['label'] = df_olid['subtask_a'].map({'OFF': 1, 'NOT': 0})
        df_olid = df_olid[['tweet', 'label']].rename(columns={'tweet': 'text'})
        datasets.append(df_olid)
        print(f"Loaded {len(df_olid)} OLID samples")
    except Exception as e:
        print(f"Could not load OLID: {str(e)}")

    try:
        print("Trying to load SBIC...")
        sbic = load_dataset("allenai/social_bias_frames", split="train", trust_remote_code=True)
        df_sbic = pd.DataFrame(sbic)

        df_sbic['label'] = df_sbic['offensiveYN'].apply(lambda x: 1 if x in ['1.0', '0.5'] else 0)

        df_sbic = df_sbic[['post', 'label']]
        df_sbic.rename(columns={'post': 'text'}, inplace=True)

        df_sbic.to_csv("sbic_dataset.csv", index=False, encoding="utf-8")

        datasets.append(df_sbic)
        print(f"Loaded and saved {len(df_sbic)} SBIC samples to 'sbic_dataset.csv'")
    except Exception as e:
        print(f"Could not load SBIC: {str(e)}")

    if datasets:
        df = pd.concat(datasets, ignore_index=True).dropna().drop_duplicates()
        df['text'] = df['text'].astype(str)

        df['text'] = df['text'].apply(clean_text)
        df['label'] = df['label'].astype(int)
        df = df[df['text'].str.len() > 5]

        print("Applying data augmentation...")
        augmented_rows = []

        for _, row in df.iterrows():
            try:
                aug_text = synonym_augment(row['text'])
                aug_text = clean_text(aug_text)
                if len(aug_text) > 5 and aug_text != row['text']:
                    augmented_rows.append({'text': aug_text, 'label': row['label']})
            except Exception as e:
                continue

        df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
        print(f"Added {len(augmented_rows)} augmented rows.")
        df_augmented.to_csv('processed_data_final.csv', index=False)
        print("DataFrame with augmented text has been saved to 'processed_data_augmented.csv'")
        return df_augmented


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = strip_leading_punctuation(text)

    text = basic_clean_text(text)

    text = contractions.fix(text)

    text = re.sub(r"<\s*user\s*>|<\s*url\s*>", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r'(^[=+\-*/]+\s*)|(\s*[=+\-*/]+\s*)+', ' ', text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(https?[:\s]?[/\\]*\s*\S+)|www\.\S+|url", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?\s+\S+|www\s+\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\w+\s+(com|org|net|gov|edu)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = ' '.join([reduce_repeated_letters(word) for word in text.split()])

    text = basic_clean_text(text)

    text = contractions.fix(text)

    return text


def basic_clean_text(text):
    text = re.sub(r"\.{2,}", ".", text)

    text = re.sub(r"\by\b", "you", text)
    text = re.sub(r"\bya\b", "you", text)
    text = re.sub(r"\byo\b", "you", text)

    text = re.sub(r"\burs\b", "yours", text)
    text = re.sub(r"\burself\b", "yourself", text)

    text = re.sub(r"\bre\b", "are", text)

    text = re.sub(r"\bid\b", "i would", text)
    text = re.sub(r"\bd\b", "would", text)

    text = re.sub(r"\bI\s*m\b", "I am", text)
    text = re.sub(r"\bm\b", "am", text)
    text = re.sub(r"\bdon\s*t\b", "don't", text)

    text = re.sub(r"\bdayum\b", "damn", text)
    text = re.sub(r"\bwtf\b", "what the fuck", text)
    text = re.sub(r"\bdafuk\b", "the fuck", text)
    text = re.sub(r"wtf\b", "what the fuck", text)
    text = re.sub(r"\bshiet\b", "shit", text)

    text = re.sub(r"\baf\b", "as fuck", text)
    text = re.sub(r"\btf\b", "the fuck", text)
    text = re.sub(r"\bdf\b", "the fuck", text)
    text = re.sub(r"\bboi\b", "boy", text)

    text = re.sub(r"\bidc\b", "i do not care", text)
    text = re.sub(r"\bve\b", "have", text)

    text = re.sub(r"\bdis\b", "this", text)
    text = re.sub(r"\bdat\b", "that", text)
    text = re.sub(r"\bda\b", "the", text)

    text = re.sub(r"\bta\b", "to", text)
    text = re.sub(r"\bar\b", "are", text)

    text = re.sub(r"\bcus\b", "because", text)

    text = re.sub(r"\byah\b", "yes", text)
    text = re.sub(r"\byeah\b", "yes", text)

    text = re.sub(r"\bwit\b", "with", text)

    text = re.sub(r"\bnuffin\b", "nothing", text)
    text = re.sub(r"\blil\b", "little", text)

    return text
