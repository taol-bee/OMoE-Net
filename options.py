import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['gsn', 'sp', 'jpeg', 'gb', 'mb'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--use_offline_dataset', action='store_true', help='Use offline degraded dataset')
parser.add_argument('--offline_dir', type=str, default='data/Train/', help='Root directory for HR and LR')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument("--wblogger",type=str,default="OMoE-Net",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="OMoE-Net",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1, help = "Number of GPUs to use for training")
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pre-trained model checkpoint')
parser.add_argument('--freeze_encoder', action='store_true', help='Whether to freeze encoder during transfer')
parser.add_argument('--run_name', type=str, default='default_run', help='Name of this training run (used for logging, saving)')
parser.add_argument('--freeze_ratio', type=float, default=1.0, help='Freeze encoder rate')

parser.add_argument('--num_experts', type=int, default=5, help='Number of experts in MoE (default: 2 for GSN and SP)')
parser.add_argument('--k_experts', type=int, default=1, help='Number of top experts to select for each sample')
parser.add_argument('--expert_layers', type=int, default=2, help='Number of transformer layers in each expert')

parser.add_argument('--orthogonal_loss_weight', type=float, default=0.1, 
                    help='Weight for orthogonal loss between expert features')
parser.add_argument('--orthogonal_eps', type=float, default=1e-8, 
                    help='Epsilon for numerical stability in orthogonal loss')

parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Gradient clipping value')


options = parser.parse_args()

