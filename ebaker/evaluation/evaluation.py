import logging
import os
import json
from training.distributed import is_master
from .linear_eval import linear_eval
from .zero_shot import zero_shot_eval
from .retrieval import retrieval_evaluation
# from .analyze_features import analyze_features
# from .sts_evaluation import sts_benchmark
from .nlp_evaluations import nlp_eval
from .wise_ft import get_wise_ft_model

try:
    import wandb
except ImportError:
    wandb = None

def evaluate(model, epoch, preprocess, args, tb_writer=None):
    if args.distributed and not is_master(args):
        return
    logging.info( f"Starting evaluation of [{args.name}] at epoch {epoch}")


    if args.eval_with_wise_ft !=1:
        logging.info( f"Perform Wise-FT evaluation with alpha={args.eval_with_wise_ft}")
        model = get_wise_ft_model(model, args, alpha=args.eval_with_wise_ft)
        distributed = args.distributed
        args.distributed = False

    if args.model_ema:
        distributed = args.distributed
        args.distributed = False
    
    linear_eval_datasets = ['CIFAR10']
    zeroshot_datasets = ['ImageNet']
    args.evaluation_workers = 8


    model.eval()
    all_metrics1 = {}
    all_metrics2 = {}
    all_metrics3 = {}
    # Image-text retrieval
    args.retrieval_data =args.retrieval_data1
    retrieval_metrics1 = retrieval_evaluation(model, epoch, preprocess, args)
    all_metrics1.update(retrieval_metrics1)
    logging.info( f"Finished evaluation1 of [{args.name}] at epoch {epoch}\n" + "\n".join([f"\t{k}\t{v}" for k, v in all_metrics1.items()]))

    args.retrieval_data =args.retrieval_data2
    retrieval_metrics2 = retrieval_evaluation(model, epoch, preprocess, args)
    all_metrics2.update(retrieval_metrics2)
    logging.info( f"Finished evaluation2 of [{args.name}] at epoch {epoch}\n" + "\n".join([f"\t{k}\t{v}" for k, v in all_metrics2.items()]))

    args.retrieval_data =args.retrieval_data3
    retrieval_metrics3 = retrieval_evaluation(model, epoch, preprocess, args)
    all_metrics3.update(retrieval_metrics3)
    logging.info( f"Finished evaluation3 of [{args.name}] at epoch {epoch}\n" + "\n".join([f"\t{k}\t{v}" for k, v in all_metrics3.items()]))   
        
        # for name, val in metrics.items():
    #     if tb_writer is not None:
    #         tb_writer.add_scalar(f"eval_retrieval/{name}", val, epoch)
    #     if args.wandb:
    #         wandb.log({f"eval_retrieval/{name}": val, 'epoch': epoch})         
    if args.save_logs:
        with open(os.path.join(args.logs, args.name, "results.jsonl"), "a+") as f:
            f.write(json.dumps(all_metrics1))
            f.write("\n")
            
            
    if args.eval_with_wise_ft !=1 or args.model_ema:
        args.distributed = distributed
        
    return all_metrics1
