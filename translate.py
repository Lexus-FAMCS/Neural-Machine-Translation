import argparse
import transformer
import torch
from tokenizers import Tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate translations using a trained Transformer model.')
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained Transformer model checkpoint")

    parser.add_argument("--tokenizers_path", type=str, required=True,
                        help="Path to the tokenizers directory containing 'en_tokenizer.json' and 'ru_tokenizer.json'")
    
    parser.add_argument("--model_max_len", type=int, default=64,
                        help="Maximum length of the model input sequences.")
    
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_hid", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling during generation. Higher values lead to more random outputs.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling threshold. Only the most probable tokens with cumulative probability less than top_p are considered during generation.")

    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save the generated translations")

    args = parser.parse_args()

    en_tokenizer = Tokenizer.from_file(f'{args.tokenizers_path}/en_tokenizer.json')
    ru_tokenizer = Tokenizer.from_file(f'{args.tokenizers_path}/ru_tokenizer.json')
    translator = transformer.Transformer(
        en_tokenizer=en_tokenizer, ru_tokenizer=ru_tokenizer,
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_hid=args.d_hid,
        dropout=args.dropout, max_len=args.model_max_len,
        device=args.device, logger=None
    ).to(args.device)
    
    translator.load_state_dict(torch.load(args.model_path, map_location=args.device))
    translator.eval()


    try:
        while True:
            en_sentence = input("\033[92mEnter an English sentence \033[0m(Ctrl+C to quit): ")
            ru_generated = translator.generate(
                en_sentence,
            )
            print(f"\033[91mGenerated Russian translation:\033[0m {ru_generated}\n")
            if args.output_file is not None:
                with open(args.output_file, 'a') as f:
                    f.write(f"Input: {en_sentence}\n")
                    f.write(f"Translation: {ru_generated}\n\n")
    except KeyboardInterrupt:
        print("\nExiting the translation script.")
