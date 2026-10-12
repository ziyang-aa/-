"""Greedy inference for the Transformer caption decoder."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from models import Encoder, TransformerDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--word-map', default=None)
    parser.add_argument('--max-len', type=int, default=30)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_map_path = Path(args.word_map or root / 'data' / 'processed' /
                         'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
    with open(word_map_path, encoding='utf-8') as f:
        word_map = json.load(f)
    rev_map = {v: k for k, v in word_map.items()}

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    decoder = TransformerDecoder(vocab_size=len(word_map)).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    encoder = Encoder().to(device).eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = Image.open(args.img).convert('RGB').resize((256, 256))
    tensor = transforms.ToTensor()(image)
    tensor = normalize(tensor).unsqueeze(0).to(device)
    start = word_map['<start>']
    end = word_map['<end>']
    tokens = [start]
    attention_steps = []
    with torch.no_grad():
        encoded = encoder(tensor)
        for _ in range(args.max_len):
            captions = torch.tensor([tokens], dtype=torch.long, device=device)
            lengths = torch.tensor([[len(tokens)]], dtype=torch.long, device=device)
            scores, _, _, alphas, _ = decoder(encoded, captions, lengths)
            attention_steps.append(alphas[0, -1].detach().cpu().tolist())
            next_token = int(scores[0, -1].argmax())
            tokens.append(next_token)
            if next_token == end:
                break

    words = [rev_map.get(t, '<unk>') for t in tokens[1:]
             if t not in {word_map['<pad>'], start, end}]
    caption = ' '.join(words)
    result = {'image': str(Path(args.img).resolve()), 'caption': caption,
              'token_ids': tokens, 'device': str(device),
              'cross_attention': attention_steps}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
