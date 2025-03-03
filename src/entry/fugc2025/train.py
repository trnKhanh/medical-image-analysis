from argparse import ArgumentParser

from training.unet_trainer import UNetTrainer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-dir", default=".", type=str)
    parser.add_argument("--log-file", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data-dir", nargs="+", required=True, type=str)
    parser.add_argument("--split-dicts", default=None)
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--image-size", default=None, nargs="+", type=int)

    parser.add_argument("--num-folds", default=5, type=int)
    parser.add_argument("--valid-rate", default=0.2, type=float)

    parser.add_argument("--num-epochs", default=1000, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--start-lr", default=1e-3, type=float)
    parser.add_argument("--oversample", default=1, type=int)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-normalization", action="store_true")

    return parser.parse_args()


def train_entry():
    args = parse_args()

    trainer = UNetTrainer(
        work_path=args.work_dir,
        device=args.device,
        data_path=args.data_dir,
        data_split_dicts=args.split_dicts,
        data_num_folds=args.num_folds,
        data_valid_rate=args.valid_rate,
        data_fold="all",
        pretrained_model=args.checkpoint,
        num_classes=args.num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_freq=5,
        log_path=args.log_file,
        optimizer=args.optimizer,
        start_lr=args.start_lr,
        warmup_steps=args.warmup_steps,
        optimizer_kwargs=dict(weight_decay=args.weight_decay),
        data_oversample=args.oversample,
        data_augment=not args.no_augment,
        data_normalize=not args.no_normalization,
        num_workers=4,
        pin_memory=True,
    )
    trainer.initialize()

    trainer.run_training()
