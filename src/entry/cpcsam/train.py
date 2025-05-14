from argparse import ArgumentParser

from training.cpcsam_trainer import CPCSAMTrainer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-path", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-only", action="store_true")

    # >>> Model parameters
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--patch-size", default=512, nargs="+", type=int)
    parser.add_argument("--image-size", default=512, nargs="+", type=int)
    parser.add_argument(
        "--sam-name",
        default="vit_b_dualmask_same_prompt_class_random_large",
        type=str,
    )
    parser.add_argument("--model-ckpt", required=True, type=str)
    parser.add_argument("--lora-rank", default=4, type=int)
    parser.add_argument("--lora-ckpt", default=None, type=str)
    parser.add_argument("--promptmode", default=["point"], nargs="+", type=str)
    parser.add_argument("--dropout-rate", default=0.0, type=float)
    parser.add_argument(
        "--num-points-prompt", default=[1, 2], type=int, nargs="+"
    )
    parser.add_argument(
        "--bbox-change-rate", default=[0.1, 0.2], type=float, nargs="+"
    )
    # <<< Model parameters

    # >>> Data parameters
    parser.add_argument("--dataset", default="ACDC", type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--labeled-num", default=1, type=int)
    parser.add_argument("--do-augment", action="store_true")
    parser.add_argument("--do-normalize", action="store_true")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--labeled-batch-ratio", default=0.5, type=float)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    # <<< Data parameters

    # >>> Training parameters
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--num-epochs", default=10000, type=int)
    parser.add_argument("--min-iter", default=10000, type=int)
    parser.add_argument("--warmup-iter", default=5000, type=int)
    parser.add_argument("--start-lr", default=1e-3, type=int)
    parser.add_argument("--lr-scheduler", default="poly", type=str)
    parser.add_argument("--lr-warmup-iter", default=250, type=int)
    parser.add_argument("--save-freq-epoch", default=100, type=int)
    parser.add_argument("--valid-freq-iter", default=200, type=int)
    parser.add_argument("--save-metric", default="dice", type=str)
    parser.add_argument("--loss", default="dice+ce", type=str)
    parser.add_argument("--dice-weight", default=0.8, type=float)
    parser.add_argument("--loss2-weight", default=1.0, type=float)
    parser.add_argument("--loss2-weight-rampup-iter", default=0, type=int)
    parser.add_argument("--loss2-weight-rampup-interval", default=100, type=int)
    parser.add_argument("--weight-rampup-length", default=200, type=int)
    parser.add_argument(
        "--coe1", dest="consistency_weight_1", default=0.4, type=float
    )
    parser.add_argument(
        "--coe2", dest="consistency_weight_2", default=0.05, type=float
    )
    parser.add_argument("--early-stop-max-patience", default=None, type=int)
    parser.add_argument("--loss3-weight", default=0.1, type=float)
    parser.add_argument("--loss3-weight-rampup-iter", default=15000, type=int)
    parser.add_argument("--loss3-weight-rampup-interval", default=100, type=int)
    parser.add_argument("--use-contrastive-loss", action="store_true")
    parser.add_argument("--contrastive-dropout-rate", default=0.0, type=float)
    parser.add_argument("--contrastive-weight", default=0.1, type=float)
    parser.add_argument("--use-adv-loss", action="store_true")
    parser.add_argument("--adv-weight", default=1.0, type=float)

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
    lr_scheduler = args_dict.pop("lr_scheduler")
    loss = args_dict.pop("loss")
    save_metric = args_dict.pop("save_metric")

    trainer = CPCSAMTrainer(
        config=args_dict,
        optimizer_name=optimizer,
        optimizer_kwargs={},
        lr_scheduler_name=lr_scheduler,
        loss_name=loss,
        save_metric_name=save_metric,
        **args_dict,
    )
    trainer.initialize()

    if test_only:
        trainer.perform_real_test()
    else:
        trainer.run_training()
