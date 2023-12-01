import pathlib
import re

import string
import json

import torch
from tqdm import tqdm

path_to_dataset = pathlib.Path("../data/fairytale_dataset")


def load_raw_fairytales_dataset(how_many: int = 2124) -> list:
    tales = []
    for file_path in tqdm(
        path_to_dataset.glob("*"), desc="Reading files", unit=" files"
    ):
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)
            tales.append(json_data["assistant_reply"])

    return tales[:how_many]


def load_processed_fairytales_dataset_for_trans(
    how_many: int = 2124, context_size: int = 100
):
    tales = load_raw_fairytales_dataset(how_many=how_many)
    preprocessed_fairytales = []

    for tale in tqdm(tales, desc="Preprocessing"):
        preprocessed_fairytales.append(preprocess_fairytale_string(tale))

    fairytale_vocab, fairytale_reverse_vocab = generate_vocabulary(
        preprocessed_fairytales
    )
    encoded_strings = encode_list_of_strings(preprocessed_fairytales, fairytale_vocab)
    encoding_pairs, decoding_pairs = generate_dataset_of_fairytales_trans(
        encoded_strings, context_size
    )

    return encoding_pairs, decoding_pairs, fairytale_vocab, fairytale_reverse_vocab


def load_processed_fairytales_dataset_for_lstm(
    how_many: int = 2124, context_size: int = 100
):
    tales = load_raw_fairytales_dataset(how_many=how_many)
    preprocessed_fairytales = []

    for tale in tqdm(tales, desc="Preprocessing"):
        preprocessed_fairytales.append(preprocess_fairytale_string(tale))

    fairytale_vocab, fairytale_reverse_vocab = generate_vocabulary(
        preprocessed_fairytales
    )
    encoded_strings = encode_list_of_strings(preprocessed_fairytales, fairytale_vocab)
    splitted_samples = generate_dataset_of_fairytales_lstm(
        encoded_strings, context_size
    )

    return splitted_samples, fairytale_vocab, fairytale_reverse_vocab


def generate_dataset_of_fairytales_trans(encoded_strings, context_size: int = 100):
    encoding_pairs, decoding_pairs = [], []

    for tale in tqdm(encoded_strings, desc="Generating encoded pairs"):
        enc, dec = generate_pairs_of_samples(tale, context_size)
        encoding_pairs.append(enc)
        decoding_pairs.append(dec)

    encoding_pairs = [item for sublist in encoding_pairs for item in sublist]
    decoding_pairs = [item for sublist in decoding_pairs for item in sublist]
    return encoding_pairs, decoding_pairs


def generate_dataset_of_fairytales_lstm(encoded_strings, context_size: int = 100):
    splitted_strings = []

    for tale in tqdm(encoded_strings, desc="Generating encoded pairs"):
        splt = generate_splitted_samples(tale, context_size)
        splitted_strings.append(splt)

    splitted_strings = [item for sublist in splitted_strings for item in sublist]
    return splitted_strings


def preprocess_and_encode_string(text, vocab):
    text = preprocess_fairytale_string(text)
    encoded_sentence = [vocab.get(token, vocab["<unk>"]) for token in text]

    return encoded_sentence


def encode_list_of_strings(text, vocab):
    encoded_sentences = []
    for sentence in tqdm(text, desc="Converting strings to integers"):
        lil_sentence = [vocab.get(token, vocab["<unk>"]) for token in sentence]
        encoded_sentences.append(lil_sentence)
    return encoded_sentences


def decode_string(text, reverse_vocab):
    decoded_sentence = []
    for token in text:
        if not token:
            continue
        else:
            decoded_sentence.append(reverse_vocab[str(token)][0])

    return decoded_sentence


def generate_pairs_of_samples(encoded_words: list, context_size: int):
    encoding_pairs = []
    decoding_pairs = []

    front_zeros = [0] * (context_size - 1)
    back_zeros = [0] * context_size

    padded_encoded_words = front_zeros + encoded_words + back_zeros

    for i in range(len(padded_encoded_words) - context_size):
        encoding_pair = padded_encoded_words[i : i + context_size]
        decoding_pair = padded_encoded_words[i + 1 : i + 1 + context_size]

        encoding_pairs.append(encoding_pair)
        decoding_pairs.append(decoding_pair)

    return encoding_pairs, decoding_pairs


def generate_splitted_samples(encoded_words: list, context_size: int):
    splitted_samples = []

    front_zeros = [0] * (context_size - 1)
    back_zeros = [0] * context_size

    padded_encoded_words = front_zeros + encoded_words + back_zeros

    for i in range(0, (len(padded_encoded_words) - context_size)):
        splitted_sample = padded_encoded_words[i : i + context_size]
        splitted_samples.append(splitted_sample)

    return splitted_samples


def preprocess_fairytale_string(fairytale: str):
    beginning_token = "<bos> "
    end_sentence_token = " <eos>"
    punctuation_without_brackets = "".join(
        c for c in string.punctuation if c not in "<>"
    )

    modified_string = beginning_token + fairytale
    modified_string = re.sub(r"\.+", ".", modified_string)
    modified_string = modified_string.replace(
        ".", f"{end_sentence_token} {beginning_token}"
    )
    modified_string = modified_string.replace("\n", " ")
    modified_string = modified_string.replace("  ", " ")
    modified_string = modified_string.lower()
    translator = str.maketrans("", "", punctuation_without_brackets)
    modified_string = modified_string.translate(translator)

    if modified_string.endswith(" <bos> "):
        modified_string = modified_string[: -len(" <bos> ")]

    if not modified_string.endswith("<eos>"):
        modified_string = modified_string + " <eos>"

    list_of_strings = modified_string.split()

    return list_of_strings


def shuffle(src_data, tgt_data):
    num_elements = src_data.size(0)
    random_indices = torch.randperm(num_elements)

    return src_data[random_indices], tgt_data[random_indices]


def pad_zeros(encoded_sentence: list, context_length: int = 100, front: bool = False):
    how_many_to_pad = context_length - len(encoded_sentence)

    if front:
        encoded_sentence = encoded_sentence[::-1]

    for i in range(how_many_to_pad):
        encoded_sentence.append(0)

    if front:
        encoded_sentence = encoded_sentence[::-1]

    return encoded_sentence


def generate_vocabulary(sentences_list):
    tokenized = [item for sublist in sentences_list for item in sublist]
    specials = [
        "<pad>",  # : 0
        "<unk>",  # : 1
        "<bos>",  # : 2
        "<eos>",  # : 3
    ]

    vocab = dict()

    for special_token in specials:
        if special_token not in vocab.keys():
            vocab[special_token] = len(vocab)

    for token in tokenized:
        if token not in vocab.keys():
            vocab[token] = len(vocab)

    reverse_vocab = dict()
    for key, val in vocab.items():
        if val not in reverse_vocab:
            reverse_vocab[int(val)] = [key]

    return vocab, reverse_vocab


def save_experiment_input(src_data, tgt_data, vocab, reverse_vocab, experiment_number):
    path = pathlib.Path(f"./models/{experiment_number}")
    path.mkdir(parents=True, exist_ok=True)

    torch.save(src_data, path / "src_data_input.pt")
    torch.save(tgt_data, path / "tgt_data_input.pt")

    with open(path / "vocab.json", "w") as f:
        json.dump(vocab, f)

    with open(path / "reverse_vocab.json", "w") as f:
        json.dump(reverse_vocab, f)

    print(f"Saved successfully")


def get_next_folder_number(directory_path):
    numeric_folders = []

    path = pathlib.Path(directory_path)

    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_dir() and item.name.isdigit():
                numeric_folders.append(item.name)

        numeric_folders.sort(key=int, reverse=True)  # Sort the folders numerically

    return int(numeric_folders[0]) + 1


def postprocess_string(list_of_strings):
    formatted_text = []
    for word in list_of_strings:
        if word == "<bos>":
            capitalize_next = True
        elif word == "<eos>":
            formatted_text.append(". ")
        elif word != "<unk>":
            if capitalize_next:
                formatted_text.append(word.capitalize())
                capitalize_next = False
            else:
                formatted_text.append(word)

    formatted_text = " ".join(formatted_text)
    formatted_text = formatted_text.replace(" .", ".")
    formatted_text = formatted_text.replace("  ", " ")

    return formatted_text
