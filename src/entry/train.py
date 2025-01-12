from argparse import ArgumentParser

from training.unet_trainer import UNetTrainer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-dir", default=".", type=str)
    parser.add_argument("--log-file", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data-dir", required=True, type=str)

    parser.add_argument("--num-epochs", default=1000, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
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
        data_num_folds=5,
        data_fold="all",
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_freq=5,
        log_path=args.log_file,
        optimizer=args.optimizer,
        start_lr=args.start_lr,
        data_oversample=args.oversample,
        data_augment=not args.no_augment,
        data_normalize=not args.no_normalization,
    )
    trainer.initialize()

    trainer.run_training()
