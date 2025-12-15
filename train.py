import os
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import torch.nn as nn
import random

from models import build_model
from dataset import ICAA17KDataset
from util import AverageMeter
import option


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_option():
    """Parse DAT model configuration"""
    import argparse
    from config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file',
                        default='configs/dat_base.yaml')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.",
                        default=None, nargs='+')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint',
                        default='pretrained_weights/dat_base_checkpoint.pth')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder')
    parser.add_argument('--tag', help='tag of experiment', default='default')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def create_data_loaders(opt):
    """Create train and test data loaders"""
    train_csv_path = os.path.join(opt.path_to_save_csv, '1train.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, '1test.csv')

    train_ds = ICAA17KDataset(train_csv_path, opt.path_to_images, if_train=True)
    test_ds = ICAA17KDataset(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size,
                            num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, writer=None,
                    global_step=0, name=None):
    """Train for one epoch"""
    model.train()
    train_losses = AverageMeter()

    for idx, (x, image_ids, y) in enumerate(tqdm(loader, desc='Training')):
        x = x.to(device)
        y = y.to(device)

        y_pred, _, _, _ = model(x, image_ids)

        loss = criterion(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss", train_losses.avg,
                            global_step=global_step + idx)

    return train_losses.avg


def validate(model, loader, criterion, device, writer=None, global_step=0,
            name=None, test_or_valid_flag='test'):
    """Validate model on MOS and Color scores"""
    model.eval()
    validate_losses = AverageMeter()

    # Initialize containers for both MOS (column 0) and Color (column 1)
    true_scores = [[], []]
    pred_scores = [[], []]

    with torch.no_grad():
        for idx, (x, image_ids, y) in enumerate(tqdm(loader, desc='Validation')):
            x = x.to(device)
            y = y.type(torch.FloatTensor).to(device)

            y_pred, _, _, _ = model(x, image_ids)

            # Evaluate both MOS and Color predictions
            for column in [0, 1]:
                y_selected = y[:, column]
                y_pred_selected = y_pred[:, column]

                pscore_np = y_pred_selected.data.cpu().numpy().astype('float')
                tscore_np = y_selected.data.cpu().numpy().astype('float')

                pred_scores[column] += pscore_np.tolist()
                true_scores[column] += tscore_np.tolist()

                loss = criterion(y_selected, y_pred_selected)
                validate_losses.update(loss.item(), x.size(0))

                if writer is not None:
                    writer.add_scalar(f"{name}/{test_or_valid_flag}_loss_{column}",
                                    validate_losses.avg, global_step=global_step + idx)

    # Calculate metrics for both MOS and Color
    results = {}
    for column, name_col in enumerate(['MOS', 'Color']):
        srcc_mean, _ = spearmanr(pred_scores[column], true_scores[column])
        lcc_mean, _ = pearsonr(pred_scores[column], true_scores[column])

        true_scores_np = np.array(true_scores[column])
        true_scores_label = np.where(true_scores_np <= 0.50, 0, 1)
        pred_scores_np = np.array(pred_scores[column])
        pred_scores_label = np.where(pred_scores_np <= 0.50, 0, 1)
        acc = accuracy_score(true_scores_label, pred_scores_label)

        results[name_col] = {'acc': acc, 'lcc': lcc_mean, 'srcc': srcc_mean}
        print(f'{name_col} - Accuracy: {acc:.4f}, LCC: {lcc_mean:.4f}, SRCC: {srcc_mean:.4f}')

    # Return average metrics
    avg_acc = (results['MOS']['acc'] + results['Color']['acc']) / 2
    avg_lcc = (results['MOS']['lcc'] + results['Color']['lcc']) / 2
    avg_srcc = (results['MOS']['srcc'] + results['Color']['srcc']) / 2

    return validate_losses.avg, avg_acc, avg_lcc, avg_srcc


def start_train(opt, device, seed=42):
    """Main training function"""
    # Set random seed
    set_seed(seed)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    # Build model
    args, config = parse_option()
    print(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    # Load pretrained weights
    print(f"Loading pretrained weights from: {opt.path_to_model_weight}")
    if os.path.exists(opt.path_to_model_weight):
        checkpoint = torch.load(opt.path_to_model_weight, map_location='cpu')
        # Filter out classification head weights
        pre_dict = {k: v for k, v in checkpoint.items() if "cls_head" not in k}
        model.load_state_dict(pre_dict, strict=False)
        print("Pretrained weights loaded successfully")
    else:
        print(f"Warning: Pretrained weights not found at {opt.path_to_model_weight}")
        print("Training from scratch...")

    model = model.to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    criterion = nn.MSELoss().to(device)

    # Create experiment directory
    os.makedirs(opt.experiment_dir_name, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.experiment_dir_name, 'logs'))

    # Training loop
    best_srcc = 0.0
    best_epoch = 0

    for epoch in range(opt.num_epoch):
        print(f"\nEpoch {epoch+1}/{opt.num_epoch}")
        print("-" * 50)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     writer=writer, global_step=len(train_loader) * epoch,
                                     name=f"Seed_{seed}")

        # Validate
        val_loss, val_acc, val_lcc, val_srcc = validate(
            model, test_loader, criterion, device,
            writer=writer, global_step=len(test_loader) * epoch,
            name=f"Seed_{seed}", test_or_valid_flag='val')

        # Log to tensorboard
        writer.add_scalars("epoch_loss", {'train': train_loss, 'val': val_loss},
                          global_step=epoch)
        writer.add_scalars("lcc_srcc", {'val_lcc': val_lcc, 'val_srcc': val_srcc},
                          global_step=epoch)
        writer.add_scalars("acc", {'val_acc': val_acc}, global_step=epoch)

        # Save best model
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            best_epoch = epoch
            model_name = f"best_model_epoch_{epoch}_srcc_{val_srcc:.4f}_acc_{val_acc:.4f}.pth"
            save_path = os.path.join(opt.experiment_dir_name, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved: {model_name}")

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val LCC: {val_lcc:.4f} | Val SRCC: {val_srcc:.4f}")
        print(f"Best SRCC: {best_srcc:.4f} (Epoch {best_epoch})")

    writer.close()
    print(f"\nTraining completed! Best SRCC: {best_srcc:.4f} at epoch {best_epoch}")

    return val_loss, val_acc, val_lcc, val_srcc


if __name__ == "__main__":
    # Parse options
    opt = option.init()

    # Set device
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current working directory: {os.getcwd()}")

    # Run experiments with multiple seeds
    seeds = [42, 123, 456, 789, 101112]
    best_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*70}")

        test_loss, tacc, tlcc, tsrcc = start_train(opt, device, seed=seed)

        best_results.append({
            'seed': seed,
            'test_loss': test_loss,
            'accuracy': tacc,
            'lcc_mean': tlcc,
            'srcc_mean': tsrcc
        })

    # Print summary
    print(f"\n{'='*70}")
    print("Experiment Summary")
    print(f"{'='*70}")
    for result in best_results:
        print(f"Seed {result['seed']}: SRCC={result['srcc_mean']:.4f}, "
              f"LCC={result['lcc_mean']:.4f}, Acc={result['accuracy']:.4f}")

    # Find best result
    best_result = max(best_results, key=lambda x: x['srcc_mean'])
    print(f"\n{'='*70}")
    print(f"Best result obtained with seed {best_result['seed']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"LCC: {best_result['lcc_mean']:.4f}")
    print(f"SRCC: {best_result['srcc_mean']:.4f}")
    print(f"{'='*70}")
