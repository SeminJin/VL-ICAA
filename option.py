import argparse
import os

def init():
    parser = argparse.ArgumentParser(description="ICAA17K Training")

    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('--path_to_images', type=str,
                        default=os.path.join(project_root, 'dataset', 'ICAA17K', 'images'),
                        help='directory to images')
    parser.add_argument('--path_to_save_csv', type=str,
                        default=os.path.join(project_root, 'dataset', 'ICAA17K'),
                        help='directory to csv_folder')
    parser.add_argument('--experiment_dir_name', type=str,
                        default=os.path.join(project_root, 'experiments'),
                        help='directory to save experiments')
    parser.add_argument('--path_to_model_weight', type=str,
                        default=os.path.join(project_root, 'pretrained_weights', 'dat_base_checkpoint.pth'),
                        help='path to DAT pretrained model weight')

    # Training hyperparameters
    parser.add_argument('--init_lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--num_epoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=6, help='number of data loading workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')

    args = parser.parse_args()
    print(args)
    return args
