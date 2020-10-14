import os
import time
import shutil
import numpy as np
import torch
from tqdm import tqdm
import argparse
import datetime
import logging
import math
import random
from tqdm import tqdm
import copy
from torch.autograd import Variable
import config



from ml_batch_loader_with_categories import MlBatchLoader_with_categories

from redial_categories_batch_loader_new import DialogueBatchLoader4Categories

from autoencoder import AutoEncoder, AutoEncoderWithCategories, Embedder

# from categories_dataset import Categories_Dataset


# from Bert4Rec_model import BertConfig, BertModel, BertForMaskedLM
# from optimization import BertAdam

# set up arguments
parser = argparse.ArgumentParser()

# parser.add_argument("--output_dir", default=datetime.datetime.now().strftime("%Y.%m.%d_%H:%M"), type=str)
parser.add_argument("--output_dir", default="", type=str)
# parser.add_argument("--input_dir", default="Results_dataset_movielens_task_ratings_BS=16", type=str)
parser.add_argument("--input_dir", default="", type=str)

parser.add_argument("--train_epochs", default=100, type=int)

parser.add_argument("--num_of_workers", default=0, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--use_cuda", default=True, type=str)
parser.add_argument("--decoder_mode", default="symmetric", type=str) # symmetric, or 1_layer or fixedB

# parser.add_argument("--trainable_categories_to_movies", default="True", type=str)
parser.add_argument("--transfer_learning", default="True", type=str)


parser.add_argument("--hidden_layers", default=[], nargs='*', type=int)


parser.add_argument("--model", default="cat_aware", type=str)
# parser.add_argument("--model", default="autoencoder", type=str)

parser.add_argument("--dataset", default="redial", type=str)
# parser.add_argument("--dataset", default="movielens", type=str)

parser.add_argument("--task", default="categories", type=str)
# parser.add_argument("--task", default="ratings", type=str)

args = parser.parse_args()

# args.trainable_categories_to_movies = True  if args.trainable_categories_to_movies == "True" else False
args.transfer_learning = True if args.transfer_learning == "True" else False



args.output_dir = args.model

hidden = "_hidden"
for size in args.hidden_layers:
    hidden += "_" + str(size)
args.output_dir += hidden

args.output_dir += "_dataset_" + args.dataset
args.output_dir += "_task_" + args.task
args.output_dir += "_BS=" + str(args.batch_size)

if args.task == "ratings":
    args.output_dir += "_Decoder"
    args.output_dir += "_FIXED" if args.decoder_mode == "fixed" else "_" + args.decoder_mode

if args.input_dir != "":
    args.output_dir += "_Transfer_Learning" if args.transfer_learning else "_Finetune"
    args.output_dir += "_From_" + args.input_dir



# in case cuda is requested but not available, switch to CPU
if torch.cuda.is_available() == False:
    args.use_cuda == False

# set random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



# if the experiment directory exists already, then delete it
if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
# create output directory
os.mkdir(args.output_dir)

# set up logger
logging.basicConfig(format = '%(asctime)s:%(message)s',
                    datefmt = '%d/%m %H:%M:%S',
                    level = logging.INFO,
                    filename = os.path.join(args.output_dir, 'log.txt') )

logger = logging.getLogger(__name__)

logger.info(args)
print(args)





if args.dataset == "movielens":
    batch_loader = MlBatchLoader_with_categories(batch_size = args.batch_size, ratings01=True)
else:
    batch_loader = DialogueBatchLoader4Categories(sources="ratings", batch_size=args.batch_size)

if args.model == "cat_aware":
    # instantiate the category predictor
    model = AutoEncoderWithCategories(number_of_movies = batch_loader.n_movies, num_of_categories = len(batch_loader.categories), layer_sizes=args.hidden_layers, decoder_mode = args.decoder_mode, task = args.task, load_model_path = args.input_dir, transfer_learning = args.transfer_learning)
elif args.model == "embedder":
	model = Embedder(n_movies = batch_loader.n_movies, num_of_categories = len(batch_loader.categories), task = args.task)
else:
    model = AutoEncoder(n_movies = batch_loader.n_movies, layer_sizes = args.hidden_layers, num_of_categories = len(batch_loader.categories), load_model_path = args.input_dir)


# print(model)
# exit()

if args.use_cuda:
    model = model.cuda()
    

if args.decoder_mode == "fixed" and args.task == "ratings":
    model.set_categories_movies_matrix(batch_loader.get_categories_movies_matrix())


# setting up 
patience_count = 0
if args.task == "ratings" and args.dataset == "redial":
    checkpoint_criteria = ["hit_at_1", "hit_at_10"]
    # checkpoint_criteria = ["hit_at_1", "hit_at_10", "hit_at_100", "ratings_loss", "Av_rank"]

else:
    checkpoint_criteria = ["loss"]

checkpoint_handler = {}

for criterion in checkpoint_criteria:
    checkpoint_handler[criterion] = {"best_value" : 1e10, "epoch" : 0, "train_results" : "", "val_results" : "", "test_results" : "", "filename": "best_" + criterion + ".pickle"}



optimizer = torch.optim.Adam(model.parameters())

for epoch in range(args.train_epochs):

    # load training batches
    if args.dataset == "redial":
        batch_data, number_of_batches = batch_loader.get_batched_samples(subset = "train", batch_size = args.batch_size)
        # # print(number_of_batches)
        # batch_data, number_of_batches = batch_loader.get_batched_samples(subset = "valid", batch_size = args.batch_size)
        # # print(number_of_batches)
        # batch_data, number_of_batches = batch_loader.get_batched_samples(subset = "test", batch_size = args.batch_size)
        # print(number_of_batches)
        # exit()



    else:
        # load_categories_batch
        number_of_batches = batch_loader.n_batches["train"]

    for i in tqdm(range(number_of_batches)):
    # for i in range(number_of_batches):

        if args.dataset == "redial":
            batch = batch_data[i]

            input_vectors = batch[0]
            rating_targets = batch[1]
            category_targets = batch[2]
        else:
            batch = batch_loader.load_categories_batch(subset = "train")
            # batch = batch_loader.load_batch(subset = "train")

            input_vectors = batch["input"]        
            rating_targets = batch["target"]
            category_targets = batch["category_target"]


        if args.use_cuda:
            input_vectors = input_vectors.cuda()
            rating_targets = rating_targets.cuda()
            category_targets = category_targets.cuda()


        output = model(input_vectors)


        if args.task == "categories":
            # this is for anything trained on categories
            loss = model.calculate_batch_cat_loss(output = output, target = category_targets)
        else:
            if args.dataset == "redial":
                # this is for redial trained on ratings
                loss = model.evaluate_output_multithreaded(input = input_vectors, target = rating_targets, output = output, num_of_workers = args.num_of_workers, eval_ranking = False)
            else:
                # this is for MovieLens trained on ratings
                loss = model.calculate_batch_loss(output = output, target = rating_targets)



        optimizer.zero_grad()

        # only using the category loss for training
        loss.backward()

        optimizer.step()

    if args.task == "categories":
        # this is for anything trained on categories
        train_loss = model.get_normalized_cat_loss()
        val_loss = model.evaluate(batch_loader, subset = "valid", args = args)
        logger.info("Ep:{:0>3d}|TRAIN| RMSE:{:.4f}, |VAL| RMSE:{:.4f}".format(epoch, train_loss, val_loss))

        checkpoint_criteria_values = {"loss" : val_loss}

    else:
        if args.dataset == "redial":
            # this is for redial trained on ratings
            train_loss, train_corr_class_rate, train_hits_at_1, train_hits_at_10, train_hits_at_100, train_av_ranking = model.get_result_metrics()
            val_loss, val_corr_class_rate, val_hits_at_1, val_hits_at_10, val_hits_at_100, val_av_ranking = model.evaluate(batch_loader, subset = "valid", args = args)

            logger.info("Ep:{:0>3d}|TRAIN| RMSE:{:.4f}, H@1:{:.4f}, H@10:{:.4f}, H@100:{:.4f}, class_rate:{:.4f}, Av_R:{:.4f}, |VAL| RMSE:{:.4f}, H@1:{:.4f}, H@10:{:.4f}, H@100:{:.4f}, class_rate:{:.4f}, Av_R:{:.4f}" \
                .format(epoch, train_loss, train_hits_at_1, train_hits_at_10, train_hits_at_100, train_corr_class_rate, train_av_ranking, val_loss, val_hits_at_1, val_hits_at_10, val_hits_at_100, val_corr_class_rate, val_av_ranking))

            # se save (1 - val_hits_at_1) instead of (val_hits_at_1) because the update best metrics rutine is based on minimizing values
            checkpoint_criteria_values = {"ratings_loss": val_loss, "hit_at_1": 1 - val_hits_at_1, "hit_at_10": 1 - val_hits_at_10, "hit_at_100": 1 - val_hits_at_100, "Av_rank": val_av_ranking}
            # checkpoint_criteria_values = {"hit_at_1": 1 - val_hits_at_1}


        else:
            # this is for MovieLens trained on ratings
            train_loss = model.get_normalized_loss()
            val_loss = model.evaluate(batch_loader, subset = "valid", args = args)
            logger.info("Ep:{:0>3d}|TRAIN| RMSE:{:.4f}, |VAL| RMSE:{:.4f}".format(epoch, train_loss, val_loss))

            checkpoint_criteria_values = {"loss" : val_loss}

    improved = False

    for criterion in checkpoint_criteria:
        if checkpoint_criteria_values[criterion] < checkpoint_handler[criterion]["best_value"]:
            # reset universal patience value
            improved = True
            # update best values for this criterion
            checkpoint_handler[criterion]["best_value"] = checkpoint_criteria_values[criterion]
            checkpoint_handler[criterion]["epoch"] = epoch + 1

            if args.task == "ratings" and args.dataset == "redial":
                checkpoint_handler[criterion]["train_results"] = "Ep:{:0>3d}|TRAIN| RMSE:{:.4f}, H@1:{:.4f}, H@10:{:.4f}, H@100:{:.4f}, class_rate:{:.4f}, Av_R:{:.4f}, |VAL| RMSE:{:.4f}, H@1:{:.4f}, H@10:{:.4f}, H@100:{:.4f}, class_rate:{:.4f}, Av_R:{:.4f}" \
                    .format(epoch, train_loss, train_hits_at_1, train_hits_at_10, train_hits_at_100, train_corr_class_rate, train_av_ranking, val_loss, val_hits_at_1, val_hits_at_10, val_hits_at_100, val_corr_class_rate, val_av_ranking)
            else:
                checkpoint_handler[criterion]["train_results"] = "Ep:{:0>3d}|TRAIN| RMSE:{:.4f}, |VAL| RMSE:{:.4f}".format(epoch, train_loss, val_loss)

            # save best model so far to file
            torch.save(model.state_dict(), os.path.join(args.output_dir, checkpoint_handler[criterion]["filename"]))

    if improved:
        patience_count = 0
    else:
        patience_count += 1
        if patience_count == args.patience:
            break

# correct the 1- hit@1
if "hit_at_1" in checkpoint_handler:
    checkpoint_handler["hit_at_1"]["best_value"] = 1 - checkpoint_handler["hit_at_1"]["best_value"]
if "hit_at_10" in checkpoint_handler:
    checkpoint_handler["hit_at_10"]["best_value"] = 1 - checkpoint_handler["hit_at_10"]["best_value"]
if "hit_at_100" in checkpoint_handler:
    checkpoint_handler["hit_at_100"]["best_value"] = 1 - checkpoint_handler["hit_at_100"]["best_value"]

logger.info("Training terminated")

for criterion in checkpoint_criteria:
    # load best model for that criterion
    model.load_state_dict(torch.load(os.path.join(args.output_dir, checkpoint_handler[criterion]["filename"])))
    # evaluate model on test set


    test_output = model.evaluate(batch_loader, subset = "test", args = args)

    if args.task == "ratings" and args.dataset == "redial":
        val_loss, val_corr_class_rate, val_hits_at_1, val_hits_at_10, val_hits_at_100, val_av_ranking  = test_output

        checkpoint_handler[criterion]["test_results"] = "|TEST| RMSE:{:.4f}, H@1:{:.4f}, H@10:{:.4f}, H@100:{:.4f}, class_rate:{:.4f}, Av_R:{:.4f}" \
            .format(val_loss, val_hits_at_1, val_hits_at_10, val_hits_at_100, val_corr_class_rate, val_av_ranking)
    else:
        checkpoint_handler[criterion]["test_results"] = "|TEST| RMSE:{:.4f}".format(test_output)

    logger.info("   " + criterion + "::  Best value: {:.4f}, epoch: {:0>2d}".format(checkpoint_handler[criterion]["best_value"], checkpoint_handler[criterion]["epoch"]))
    logger.info("Train: " + checkpoint_handler[criterion]["train_results"])
    logger.info("Test : " + checkpoint_handler[criterion]["test_results"])

    logger.info("")

    if args.dataset == "movielens":
        model.save_pruned_model_for_redial(output_dir = args.output_dir)
