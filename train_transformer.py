"""Train or smoke-test the Transformer caption decoder on the prepared data."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import CaptionDataset
from models import Encoder, TransformerDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default=None)
    parser.add_argument('--data-name', default='coco_5_cap_per_img_5_min_word_freq')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='0 runs a full epoch; use 1 for a short verification')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    data_folder = Path(args.data_folder or workspace / 'data' / 'processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(data_folder / f'WORDMAP_{args.data_name}.json', encoding='utf-8') as f:
        word_map = json.load(f)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = CaptionDataset(str(data_folder), args.data_name, 'TRAIN',
                             transform=normalize)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    encoder = Encoder().to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    decoder = TransformerDecoder(vocab_size=len(word_map)).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=4e-4)
    criterion = nn.CrossEntropyLoss()

    print(f'device: {device}')
    print(f'vocabulary: {len(word_map)}; dataset captions: {len(dataset)}')
    for epoch in range(args.epochs):
        decoder.train()
        running = 0.0
        batches = 0
        for imgs, caps, caplens in loader:
            imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)
            with torch.no_grad():
                encoded = encoder(imgs)
            scores, caps_sorted, decode_lengths, _, _ = decoder(encoded, caps, caplens)
            targets = caps_sorted[:, 1:]
            packed_scores = pack_padded_sequence(scores, decode_lengths,
                                                  batch_first=True).data
            packed_targets = pack_padded_sequence(targets, decode_lengths,
                                                   batch_first=True).data
            loss = criterion(packed_scores, packed_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
            batches += 1
            print(f'epoch={epoch + 1} batch={batches} loss={loss.item():.4f}')
            if args.max_batches and batches >= args.max_batches:
                break
        print(f'epoch={epoch + 1} mean_loss={running / max(1, batches):.4f}')

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'decoder': decoder.state_dict(), 'word_map': word_map}, save_path)
        print(f'saved: {save_path}')


if __name__ == '__main__':
    main()
