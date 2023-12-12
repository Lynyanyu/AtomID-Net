from trainer import Trainer
import argparse
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data_dir', default='D:\\Projects\\Github\\AtomIDNet\\datasets',
                        help='training data directory')
    parser.add_argument('--save_dir', default='D:\\Projects\\Github\\AtomIDNet\\models',
                        help='directory to save models.')
    
    parser.add_argument('--pretrained', type=str, default="True",
                        help='initialize a pretrained model')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='the initial learning rate')
    parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[50, 100, 150],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max_epoch', type=int, default=201,
                        help='max training epoch')
    parser.add_argument('--val_start', type=int, default=150,
                        help='epoch that start validation')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='the num of training process')

    parser.add_argument('--crop_size', type=int, default=256,
                        help='the crop size of the train image')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.train()
