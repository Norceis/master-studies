import argparse
from src.my_lstm_v2 import generate_fairytale_lstm_cli


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fairytale using an LSTM model."
    )
    parser.add_argument(
        "--input_text",
        type=str,
        help="Input text for generating the fairytale",
        required=True,
    )
    parser.add_argument(
        "--context_len", type=int, default=50, help="Context length for the fairytale"
    )
    parser.add_argument(
        "--how_many_sentences",
        type=int,
        default=10,
        help="Number of sentences in the fairytale",
    )
    parser.add_argument(
        "--experiment_number", type=int, default=3, help="Experiment number"
    )
    parser.add_argument(
        "--model_name", type=str, default="model_after_160_epoch.pth", help="Model name"
    )

    args = parser.parse_args()

    generated_fairytale = generate_fairytale_lstm_cli(
        input_text=args.input_text,
        context_len=args.context_len,
        how_many_sentences=args.how_many_sentences,
        experiment_number=args.experiment_number,
        model_name=args.model_name,
    )

    print(generated_fairytale)


if __name__ == "__main__":
    main()
