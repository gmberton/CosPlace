
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import parser
import commons
import cosface_loss
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
best_model = args.best_model #path to best model: "AML23-Cosplace/model/results/best_..."
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

model = model.to(args.device).train()

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Test set: {test_ds}")

#### Test best model on test set
best_model_state_dict = torch.load(best_model)
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")
