import sys
import parser
import logging
import commons
from datetime import datetime
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

start_time = datetime.now()
commons.make_deterministic()
logging.info(" ".join(sys.argv))
args = parser.parse_arguments()


#### Datasets
groups = [TrainDataset(args, args.train_set_folder) for n in range(args.groups_num)]

for g in groups:
    print(g.get__class_id__group_id)