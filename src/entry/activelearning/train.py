from argparse import ArgumentParser

from training.al_trainer import ALTrainer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-path", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--resume", default=None, type=str)

    # >>> Model parameters
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--block-type", default="plain", type=str)
    parser.add_argument("--block-normalization", default="batch", type=str)
    parser.add_argument("--dropout-prob", default=0.1, type=float)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--ds-layer", default=3, type=int)
    parser.add_argument("--patch-size", default=256, nargs="+", type=int)
    parser.add_argument("--image-size", default=256, nargs="+", type=int)
    parser.add_argument("--model-ckpt", default=None, type=str)
    # <<< Model parameters

    # >>> Data parameters
    parser.add_argument("--dataset", default="ACDC", type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--do-augment", action="store_true")
    parser.add_argument("--do-normalize", action="store_true")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    # <<< Data parameters

    # >>> Training parameters
    parser.add_argument("--valid-mode", default="volumn", type=str)
    parser.add_argument("--num-rounds", default=5, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--active-selector", default="random", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--num-iters", default=4000, type=int)
    parser.add_argument("--start-lr", default=1e-3, type=int)
    parser.add_argument("--lr-scheduler", default="poly", type=str)
    parser.add_argument("--lr-warmup-iter", default=250, type=int)
    parser.add_argument("--save-freq-epoch", default=100, type=int)
    parser.add_argument("--valid-freq-iter", default=200, type=int)
    parser.add_argument("--save-metric", default="dice", type=str)
    parser.add_argument("--loss", default="dice+ce", type=str)
    parser.add_argument("--dice-weight", default=1.0, type=float)
    parser.add_argument("--ce-weight", default=1.0, type=float)
    parser.add_argument("--early-stop-max-patience", default=None, type=int)
    # <<< Training parameters

    # >>> Log parameters
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.add_argument("--log-path", default=None, type=str)
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--exp-name", default="", type=str)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-api-key", default=None, type=str)
    # <<< Log parameters

    return parser.parse_args()


def train_entry():
    args = parse_args()
    args_dict = vars(args)
    test_only = args_dict.pop("test_only")
    optimizer = args_dict.pop("optimizer")
    weight_decay = args_dict.pop("weight_decay")
    lr_scheduler = args_dict.pop("lr_scheduler")
    loss = args_dict.pop("loss")
    save_metric = args_dict.pop("save_metric")
    active_selector = args_dict.pop("active_selector")

    args_dict["optimizer_name"] = optimizer
    args_dict["optimizer_kwargs"] = {"weight_decay": weight_decay}
    args_dict["lr_scheduler_name"] = lr_scheduler
    args_dict["loss_name"] = loss
    args_dict["save_metric_name"] = save_metric
    args_dict["active_selector_name"] = active_selector

    trainer = ALTrainer(
        config=args_dict,
        **args_dict,
    )
    trainer.initialize()

    if test_only:
        trainer.perform_real_test()
    else:
        trainer.run_training()
