"""``paintdetect`` command-line interface.

Subcommands:
    train     Train (and optionally evaluate) from a YAML config.
    predict   Run inference on one or more images with a saved model.
    evaluate  Score a model on the config's test set.
    export    Convert any legacy checkpoint into the HF save_pretrained format.
"""

import argparse
import logging

import torch
from PIL import Image


def _cmd_train(args):
    from .train import train_model
    train_model(args.config, run_eval=args.eval)


def _cmd_predict(args):
    from .serialization import load_model
    from .inference import predict_img, mask_to_image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    net = load_model(args.model, map_location=device)
    net.to(device=device)

    mask_values = [0, 1]
    outputs = args.output or [f'{f.rsplit(".", 1)[0]}_OUT.png' for f in args.input]
    for filename, out_filename in zip(args.input, outputs):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        mask = predict_img(net=net, full_img=img, scale_factor=args.scale,
                           out_threshold=args.mask_threshold, device=device)
        if not args.no_save:
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')


def _cmd_evaluate(args):
    from .config import load_config
    from .serialization import load_model
    from .inference import run_test_eval

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(args.model, map_location=device)
    net.to(device=device)
    run_test_eval(net, config, device)


def _cmd_export(args):
    from .serialization import load_model, save_model
    net = load_model(args.model)
    out_dir = save_model(net, args.out)
    print(f'Exported model to {out_dir} (config.json + model.safetensors)')


def build_parser():
    parser = argparse.ArgumentParser(prog='paintdetect', description='Bee image U-Net segmentation')
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train from a YAML config')
    p_train.add_argument('--config', '--config_file', dest='config', required=True,
                         help='YAML experiment config')
    p_train.add_argument('--eval', action='store_true', help='Also run test-set evaluation after training')
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser('predict', help='Run inference on images')
    p_pred.add_argument('--model', '-m', required=True,
                        help='Model: save_pretrained dir, legacy .pth, or HF repo id')
    p_pred.add_argument('--input', '-i', nargs='+', required=True, help='Input image filenames')
    p_pred.add_argument('--output', '-o', nargs='+', help='Output mask filenames')
    p_pred.add_argument('--no-save', '-n', action='store_true', help='Do not save output masks')
    p_pred.add_argument('--mask-threshold', '-t', type=float, default=0.5)
    p_pred.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for input images')
    p_pred.set_defaults(func=_cmd_predict)

    p_eval = sub.add_parser('evaluate', help='Score a model on the config test set')
    p_eval.add_argument('--config', '--config_file', dest='config', required=True)
    p_eval.add_argument('--model', '-m', required=True)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_exp = sub.add_parser('export', help='Convert a checkpoint to HF save_pretrained format')
    p_exp.add_argument('--model', '-m', required=True, help='Legacy .pth or existing model')
    p_exp.add_argument('--out', '-o', required=True, help='Output directory')
    p_exp.set_defaults(func=_cmd_export)

    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
