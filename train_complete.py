import os
import shutil
import csv
import random
from tqdm import tqdm, trange
from io import open
import jsonlines
from operator import add
# from functools import reduce
import datetime
import re
import nltk
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import re
import pickle
import math

import transformers

from redial_categories_batch_loader import DialogueBatchLoader4Transformers
from complete_model import Complete_Model
from category_pref_model import Cat_Pref_BERT


def main():
	# set up arguments
	parser = argparse.ArgumentParser()
	# model parameters
	parser.add_argument('--output_dir', type=str, default='')
	# dataset parameters
	parser.add_argument('--redial_dataset_path', type=str, default='Datasets/redial')
	parser.add_argument('--train_filename', type=str, default='train_data')
	parser.add_argument('--val_filename', type=str, default='valid_data')
	parser.add_argument('--test_filename', type=str, default='test_data.jsonl')
	parser.add_argument('--movie_details_filename', type=str, default='movie_details.csv')
	parser.add_argument('--conversation_length_limit', type=int, default=40)
	parser.add_argument('--utterance_length_limit', type=int, default=80)
	parser.add_argument("--input_length_limit", default=1024, type=int)

	parser.add_argument("--learning_rate", default=1e-4, type=float,
						help="The initial learning rate for Adam.")

	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")


	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument("--num_train_epochs", default=100, type=int,
						help="Total number of training epochs to perform.")


	parser.add_argument("--conversations_per_batch", default=1, type=int, help="How many conversations should a batch contain.")
	parser.add_argument("--max_samples_per_gpu", default=1, type=int,
						help="If != -1: Splits a batch into minibatches of size equal to this. Was implemented in order to fit large models in small GPUs. Practically, it does not affect training!")

	parser.add_argument('--seed', type=int, default=42)

	parser.add_argument("--use_cuda", default="True", type=str)

	parser.add_argument("--finetune", default="False", type=str, help="Whether to finetune the text-base model, or to freeze its parameters.")
	parser.add_argument("--use_ground_truth", default="True", type=str, help="Whether to use the predicted SA/Cat prediction, or the Ground Truth values.")


	parser.add_argument("--base_model_dir", default="Category_Preference_Model", type=str)

	parser.add_argument("--reproduce", default="Ours", type=str, help="Choose which model to reproduce among: {ours, ours_gt, e2e}")
	
	args = parser.parse_args()

	args.use_cuda = True if args.use_cuda == "True" and torch.cuda.is_available() else False
	args.finetune = True if args.finetune == "True" else False
	args.use_ground_truth = True if args.use_ground_truth == "True" else False

	if args.reproduce == "ours":
		output_dir = "Complete_Model-Ours"
		# do not update the parameters of the category preference model
		args.finetune = False
		args.use_ground_truth = False
		if os.path.isfile( os.path.join(args.base_model_dir, "best.pickle" )) == False:
			print("The trained Category Preference Prediction model, could not be found in direcotry:\n" + args.base_model_dir)
			print("Please set the parammeter \"--base_model_dir\" to the directory of the trained model")
			exit()

	elif args.reproduce == "ours_gt":
		output_dir = "Complete_Model-Ours_GT"
		# Use the Ground Truth values of the target item's categories, and do not use any Category preference prediction model
		args.use_ground_truth = True
		args.base_model_dir = ""
	elif args.reproduce == "e2e":
		output_dir = "Complete_Model-E2E"
		# "load" a pretrained BERT, use it as an initiation for the category preference prediction module, and train the complete model (including bert) for the target item recommendation task.
		args.finetune = True
		args.base_model_dir = ""
		args.use_ground_truth = False
	else:
		print("The \"--reproduce\" parameter was not correctly specified!\nPlease choose among {ours, ours_gt, e2e}, and type it lower case.")
		exit()


	if args.output_dir == "":
		args.output_dir = output_dir

	if os.path.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)
	# create output directory
	os.mkdir(args.output_dir)


	log_filename =  "log.txt"

	# set up logging
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt = '%m/%d/%Y %H:%M:%S',
						level = logging.INFO,
						filename = os.path.join(args.output_dir, log_filename) )
	logger = logging.getLogger(__name__)

	logging.getLogger(transformers.__name__).setLevel(logging.ERROR)
	logging.getLogger(nltk.__name__).setLevel(logging.ERROR)

	logger.info(args)
	print(args)


	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# instanciate batch loader
	batch_loader = DialogueBatchLoader4Transformers(conversations_per_batch=args.conversations_per_batch, \
	 max_input_length = args.input_length_limit, conversation_length_limit = args.conversation_length_limit, \
	 utterance_length_limit = args.utterance_length_limit, data_path=args.redial_dataset_path, \
	 train_path=args.train_filename, valid_path=args.val_filename, test_path=args.test_filename, \
	 movie_details_path=args.movie_details_filename)

	model = Complete_Model(vocab_size = batch_loader.vocabulary_size, cat_size = len(batch_loader.categories), n_movies = batch_loader.n_movies, args = args)

	if args.use_cuda:
		model = model.cuda()

	best_validation_loss = 1e10
	patience_count = 0

	for epoch in range(args.num_train_epochs):

		# train model
		train_metrics = model.train_epoch(batch_loader = batch_loader)
		# evaluate model
		eval_metrics = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

		log_text = "Epoch {:0>2d}: | TRAIN | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} | EVAL | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} " \
			.format(epoch, train_metrics["Loss"], train_metrics["Hit@1"], train_metrics["Hit@10"], train_metrics["Hit@100"], eval_metrics["Loss"], eval_metrics["Hit@1"], eval_metrics["Hit@10"], eval_metrics["Hit@100"])

		logger.info(log_text)

		validation_loss = eval_metrics["Loss"]

		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			patience_count = 0
			torch.save(model.state_dict(), os.path.join(args.output_dir, "best_complete.pickle"))
		else:
			patience_count += 1
			if patience_count == args.patience:
				logger.info("Training terminated because maximum patience of {}, was exceeded".format(args.patience))
				break

	model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_complete.pickle")))
	test_metrics = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

	logger.info("| TEST | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} ".format(test_metrics["Loss"], test_metrics["Hit@1"], test_metrics["Hit@10"], test_metrics["Hit@100"]))



if __name__ == '__main__':
	main()





