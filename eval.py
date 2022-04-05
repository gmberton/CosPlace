
import sys
import torch
import logging
import multiprocessing
from datetime import datetime
torch.backends.cudnn.benchmark= True  # Provides a speedup

import test
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model != None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device)

test_ds = TestDataset(args.dataset_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str[:20]}")

