import os
import json
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import (
    save_checkpoint, adjust_learning_rate, clip_gradient,
    AverageMeter, accuracy
)

from nltk.translate.bleu_score import corpus_bleu

# Data parameters
workspace = Path(__file__).resolve().parent.parent
data_folder = str(workspace / 'data' / 'processed')
checkpoint_folder = str(workspace / 'checkpoints')
data_name   = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim        = 512   # dimension of word embeddings
attention_dim  = 512   # dimension of attention linear layers
decoder_dim    = 512   # dimension of decoder RNN
dropout        = 0.5

# Training parameters
start_epoch            = 0
epochs                 = 120    # total epochs
batch_size             = 32
workers                = 0     # for data-loading; h5py + multiple workers can crash
encoder_lr             = 1e-4   # learning rate for encoder if fine-tuning
decoder_lr             = 4e-4   # learning rate for decoder
grad_clip              = 5.     # clip gradients at this value
alpha_c                = 1.     # attention regularization weight
print_freq             = 100    # print status every N batches
fine_tune_encoder      = False  # whether to unfreeze and train the encoder
checkpoint             = None   # path to existing checkpoint (if resuming)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # speeds up fixed-size inputs


def load_checkpoint_compatibility(path):
    """Load checkpoints saved with older PyTorch optimizer objects."""
    original_setstate = torch.optim.Optimizer.__setstate__

    def compatible_setstate(optimizer, state):
        if not hasattr(optimizer, 'defaults'):
            optimizer.defaults = {}
        original_setstate(optimizer, state)

    torch.optim.Optimizer.__setstate__ = compatible_setstate
    try:
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        torch.optim.Optimizer.__setstate__ = original_setstate

def main():
    global start_epoch, epochs_since_improvement, best_bleu4

    # === Log device and hyperparameters ===
    print(f"\n>>> Training on device: {device}")
    print(f"    epochs={epochs}, batch_size={batch_size}, encoder_lr={encoder_lr}, decoder_lr={decoder_lr}")
    print(f"    fine_tune_encoder={fine_tune_encoder}\n")

    # Read word map
    word_map_file = os.path.join(data_folder, f'WORDMAP_{data_name}.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)
    print(f">>> Vocabulary size: {vocab_size}\n")

    # Initialize or load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim, emb_dim, decoder_dim, vocab_size, dropout=dropout)
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr
        )
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=encoder_lr
        ) if fine_tune_encoder else None
        epochs_since_improvement = 0
        best_bleu4 = 0.
    else:
        print(f">>> Resuming from checkpoint: {checkpoint}")
        # Checkpoints contain complete local model objects, so weights_only must be disabled.
        ckpt = load_checkpoint_compatibility(checkpoint)
        start_epoch = ckpt['epoch'] + 1
        epochs_since_improvement = ckpt['epochs_since_improvement']
        best_bleu4 = ckpt['bleu-4']
        decoder = ckpt['decoder']
        decoder_optimizer = ckpt['decoder_optimizer']
        encoder = ckpt['encoder']
        encoder_optimizer = ckpt['encoder_optimizer']
        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(True)
            encoder_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr
            )

    # Move models to device
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Log parameter counts
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f">>> Encoder trainable parameters: {enc_params:,}")
    print(f">>> Decoder trainable parameters: {dec_params:,}\n")

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Data loaders
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    print(f">>> TRAIN set size: {len(train_loader.dataset)} captions")
    print(f">>>  VAL set size: {len(val_loader.dataset)} captions\n")

    # === Training loop ===
    for epoch in range(start_epoch, epochs):
        # Early stopping / LR decay logic
        if epochs_since_improvement == 20:
            print("No improvement for 20 epochs -> stopping.")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # Train for one epoch
        train_epoch(train_loader, encoder, decoder, criterion,
                    encoder_optimizer, decoder_optimizer, epoch)

        # Validate
        recent_bleu4 = validate_epoch(val_loader, encoder, decoder, criterion, word_map)
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        epochs_since_improvement = 0 if is_best else epochs_since_improvement + 1

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement,
                        encoder, decoder, encoder_optimizer, decoder_optimizer,
                        recent_bleu4, is_best, checkpoint_folder)

def train_epoch(train_loader, encoder, decoder, criterion,
                encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top5accs   = AverageMeter()

    start = time.time()
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        # Forward
        imgs_enc = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_enc, caps, caplens)

        # Prepare targets
        targets = caps_sorted[:, 1:]

        # scores, _  = pack_padded_sequence(scores, decode_lengths, batch_first=True) #####
        # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        packed_scores  = pack_padded_sequence(scores,  decode_lengths, batch_first=True)
        packed_targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores  = packed_scores.data    # shape (sum(decode_lengths), vocab_size)
        targets = packed_targets.data   # shape (sum(decode_lengths),)




        # Loss + attention reg
        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Backward
        decoder_optimizer.zero_grad()
        if encoder_optimizer: encoder_optimizer.zero_grad()
        loss.backward()

        # Clip & step
        clip_gradient(decoder_optimizer, grad_clip)
        if encoder_optimizer: clip_gradient(encoder_optimizer, grad_clip)
        decoder_optimizer.step()
        if encoder_optimizer: encoder_optimizer.step()

        # Metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top-5 Acc {top5accs.val:.3f}% ({top5accs.avg:.3f}%)')

def validate_epoch(val_loader, encoder, decoder, criterion, word_map):
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses     = AverageMeter()
    top5accs   = AverageMeter()
    references = []
    hypotheses = []

    start = time.time()
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

            # Forward
            if encoder: imgs_enc = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_enc, caps, caplens)

            # Loss
            targets = caps_sorted[:,1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            losses.update(loss.item(), sum(decode_lengths))

            # Accuracy
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(f'Validation [{i}/{len(val_loader)}]\t'
                      f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Top-5 Acc {top5accs.val:.3f}% ({top5accs.avg:.3f}%)')

            # Collect for BLEU
            allcaps = allcaps[sort_ind.cpu()]
            for j in range(allcaps.size(0)):
                img_caps = allcaps[j].tolist()
                references.append([
                    [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}]
                    for c in img_caps
                ])
            _, preds = torch.max(scores_copy, dim=2)
            preds = [preds[j,:decode_lengths[j]].tolist() for j in range(len(decode_lengths))]
            hypotheses.extend(preds)

    bleu4 = corpus_bleu(references, hypotheses)
    print(f'\n * VALIDATION: Loss {losses.avg:.3f}, Top-5 {top5accs.avg:.3f}%, BLEU-4 {bleu4:.4f}\n')
    return bleu4

if __name__ == '__main__':
    main()
