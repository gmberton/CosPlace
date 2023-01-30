import sys
import our_parser
import logging
import commons
import plots
from datetime import datetime
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


start_time = datetime.now()
commons.make_deterministic()
logging.info(" ".join(sys.argv))
args = our_parser.parse_arguments()


#### Datasets
groups = [TrainDataset(args, args.train_set_folder) for n in range(args.groups_num)]

for g in groups:
    plots.plot_histogram(g.classes_ids, g.images_per_class)