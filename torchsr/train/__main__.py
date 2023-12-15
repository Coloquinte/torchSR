from .helpers import report_model
from .options import args
from .trainer import Trainer

trainer = Trainer()

if args.validation_only or args.images:
    if (
        args.load_pretrained is None
        and args.load_checkpoint is None
        and not args.download_pretrained
    ):
        raise ValueError(
            "For validation, please use --load-pretrained CHECKPOINT or --download-pretrained"
        )
    if args.images:
        trainer.run_model()
    else:
        trainer.validation()
else:
    report_model(trainer.model)
    trainer.train()
