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

	parser.add_argument("--learning_rate", default=1e-4, type=float,
						help="The initial learning rate for Adam.")
	# parser.add_argument("--weight_decay", default=0.0, type=float,
	#                     help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")


	# parser.add_argument('--max_train_epochs', type=int, default=100)
	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument("--num_train_epochs", default=100, type=int,
						help="Total number of training epochs to perform.")


	parser.add_argument("--conversations_per_batch", default=5, type=int)
	parser.add_argument("--base_conversations_per_batch", default=1, type=int)
	parser.add_argument("--max_samples_per_gpu", default=-1, type=int,
						help="If > 0: Splits a conversation into minibatches of size equal to this. Otherwise the batch is an integer of conversations as defined in --conversations_per_batch.")


	# model parameters
	parser.add_argument("--input_length_limit", default=1024, type=int)
	# parser.add_argument("--hidden_size", default=64, type=int)
	# parser.add_argument("--num_hidden_layers", default=2, type=int)
	# parser.add_argument("--num_attention_heads", default=2, type=int)
	# parser.add_argument("--intermediate_size", default=256, type=int)

	parser.add_argument('--seed', type=int, default=42)


	# parser.add_argument("--cat_sa_alpha", default=1.0, type=float)
	# parser.add_argument("--sem_nlg_alpha", default=0.5, type=float)


	parser.add_argument("--use_cuda", default="True", type=str)



	# parser.add_argument("--use_CLS_output", default="True", type=str)
	# parser.add_argument("--CLS_mode", default="1_CLS", type=str) # CLS_mode = "1_CLS" or "C_CLS_1_linear" or "C_CLS_C_linears"
	# parser.add_argument("--HIBERT", default="False", type=str)
	# parser.add_argument("--task", default="semantic", type=str)  # task = "semantic" or "nlg" or "pretrained" or "full"
	# parser.add_argument("--use_pretrained", default="True", type=str)


	parser.add_argument("--debug_run", default="False", type=str)



	# parser.add_argument("--rec_model", default="autorec", type=str) # rec_model = "autorec" or "CAAE" or "Cat2Item"
	# parser.add_argument("--CAAE_encoder", default="False", type=str) # If True, then we train the Encoder part of the CAAE, using the Category vectors as targets. Otherwise we train the item rating Decoder
	parser.add_argument("--finetune", default="False", type=str, help="Whether to finetune the text-base model, or to freeze its parameters.")
	parser.add_argument("--use_ground_truth", default="True", type=str, help="Whether to use the predicted SA/Cat prediction, or the Ground Truth values.")


	parser.add_argument("--base_model_dir", default="Category_Preference_Model", type=str) # FLAT_semantic_1_CLS_CAT_SA_a_0.5_Hidden_256_Layers_3_Heads_4_Inter_1024_BS_1
	# parser.add_argument("--Base_Model_Class", default="", type=str) 
	# parser.add_argument("--base_model_name", default="", type=str, help = "Depending on the recommender model, we load either Best SA or CatPred performance Transformer model.")

	# parser.add_argument('--rec_layer_sizes','--list', nargs='+', default=[])

	parser.add_argument("--reproduce", default="Ours", type=str, help="Choose which model to reproduce among: {ours, ours_gt, e2e}")
	


	args = parser.parse_args()

	# args.HIBERT = True if args.HIBERT == "True" else False
	# args.use_pretrained = True if args.use_pretrained == "True" else False
	# args.use_CLS_output = True if args.use_CLS_output == "True" else False
	args.use_cuda = True if args.use_cuda == "True" and torch.cuda.is_available() else False
	args.debug_run = True if args.debug_run == "True" else False
	args.finetune = True if args.finetune == "True" else False
	args.use_ground_truth = True if args.use_ground_truth == "True" else False
	# The CAAE encoder is only True if we are using the CAAE model and training the Category Encoder
	# args.CAAE_encoder = True if args.CAAE_encoder == "True" and args.rec_model == "CAAE" else False


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


	# args.base_model_dir = base_model_dir



	# model.load_state_dict(torch.load( os.path.join(args.base_model_dir, args.base_model_name) ))

	# exit()



	# instanciate batch loader
	# batch_loader = DialogueBatchLoader4Categories(sources="dialogue_to_categories ratings", conversations_per_batch=args.conversations_per_batch, max_input_length = args.input_length_limit)
	# batch_loader = DialogueBatchLoader4Transformers(conversations_per_batch=args.conversations_per_batch, max_input_length = args.input_length_limit, HIBERT = args.HIBERT, use_pretrained = args.use_pretrained, use_1_CLS = args.use_1_CLS)


	# batch = batch_loader.load_batch(subset="train", complete = True)

	# print(batch[0].size())

	# print(batch[-1])


	# exit()


	if args.output_dir == "":
		args.output_dir = output_dir

	if args.debug_run:
		args.output_dir = os.path.join(args.base_model_dir, "debug_run")
		args.conversations_per_batch = 1
		args.max_samples_per_gpu = 1
		args.num_train_epochs = 5
		# torch.set_num_threads(4)
		# OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 train.py --HIBERT False --debug_run True --use_cuda False

		# args.use_cuda = False




	# We always renew the directory we save models and logs, except for the case we are building the CAAE Decoder
	# if not ( args.CAAE_encoder == False and args.rec_model == "CAAE"):



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
						# LOG_FILE_INFO  = os.path.join(args.output_dir, 'log.info'),
						# LOG_FILE_WARNING = os.path.join(args.output_dir, 'log.warning')):
	logger = logging.getLogger(__name__)

	logging.getLogger(transformers.__name__).setLevel(logging.ERROR)
	logging.getLogger(nltk.__name__).setLevel(logging.ERROR)

	# args.n_gpu = torch.cuda.device_count() if args.use_cuda else 1

	args.n_gpu = 1

	logger.info(args)
	print(args)


	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# if we are using GPT2, then we set the hidden size of the transformers to be equal to the GPT2 hidden size
	# if args.use_pretrained:
	#     args.hidden_size = 768


	# instanciate batch loader
	# batch_loader = DialogueBatchLoader4Categories(sources="dialogue_to_categories ratings", conversations_per_batch=args.conversations_per_batch, max_input_length = args.input_length_limit)
	batch_loader = DialogueBatchLoader4Transformers(conversations_per_batch=args.conversations_per_batch, \
	 max_input_length = args.input_length_limit, conversation_length_limit = args.conversation_length_limit, \
	 utterance_length_limit = args.utterance_length_limit, data_path=args.redial_dataset_path, \
	 train_path=args.train_filename, valid_path=args.val_filename, test_path=args.test_filename, \
	 movie_details_path=args.movie_details_filename)

	# print( batch_loader.vocabulary_size)
	# print( len(batch_loader.categories))
	# print(batch_loader.n_movies)

	# print()
	# print(os.path.join(args.base_model_dir, args.base_model_name) )
	# print()


	model = Complete_Model(vocab_size = batch_loader.vocabulary_size, cat_size = len(batch_loader.categories), n_movies = batch_loader.n_movies, args = args)

	if args.use_cuda:
		model = model.cuda()


	# batch = batch_loader.load_batch(subset="train", complete = True)

	# model(batch)

	# print(model.get_metrics_and_reset())


	# exit()

	# model.initialize_optimizer_and_scheduler(n_train_batches = batch_loader.n_batches["train"])



	# evaluate model
	# eval_metrics = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

	# print(eval_metrics)


			# metrics["Hit@1"] = self.hit_at_1_counter / float(self.total_number_of_targets)
			# metrics["Hit@10"] = self.hit_at_10_counter / float(self.total_number_of_targets)
			# metrics["Hit@100"] = self.hit_at_100_counter / float(self.total_number_of_targets)
			# metrics["Loss"] = self.loss_sum / float(self.total_number_of_targets)
# 

	# validation_loss = eval_metrics["Loss"]
# 
	# best_validatio



	best_validation_loss = 1e10
	patience_count = 0

	# train_metrics = {key: - 1 for key in eval_metrics }

	# checkpoint_handler, train_results_text, criteria_improved = update_ckeckpoint_handler(checkpoint_handler, train_metrics, eval_metrics, -1)





	# for criterion in criteria_improved:
	#     torch.save(model.state_dict(), os.path.join(args.output_dir, checkpoint_handler[criterion]["filename"]))

	# logger.info(train_results_text)


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

		# update checkpoint handler
		# checkpoint_handler, train_results_text, criteria_improved = update_ckeckpoint_handler(checkpoint_handler, train_metrics, eval_metrics, epoch)

		# logger.info(train_results_text)

		# if len(criteria_improved) != 0:
		#     patience_count = 0
		#     for criterion in criteria_improved:
		#         torch.save(model.state_dict(), os.path.join(args.output_dir, "best_complete.pickle"))
		# else:
		#     patience_count += 1
		#     if patience_count == args.patience:
		#         logger.info("Training terminated because maximum patience of {}, was exceeded".format(args.patience))
		#         break
							


	model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_complete.pickle")))
	test_metrics = model.evaluate_model(batch_loader = batch_loader, subset = "valid")

	logger.info("| TEST | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} ".format(test_metrics["Loss"], test_metrics["Hit@1"], test_metrics["Hit@10"], test_metrics["Hit@100"]))

	# # evaluate on test set

	# for criterion in checkpoint_handler:
	#     # load best model for that criterion
	#     model.load_state_dict(torch.load(os.path.join(args.output_dir, checkpoint_handler[criterion]["filename"])))

	#     # evaluate model
	#     eval_metrics = model.evaluate_model(batch_loader = batch_loader, subset = "test")


	#     # if we are currently training a recommending model
	#     if "Hit@1" in eval_metrics:

	#         checkpoint_handler[criterion]["test_results"] = "Epoch {:0>2d}: | TEST | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} " \
	#                     .format(checkpoint_handler[criterion]["epoch"], eval_metrics["Loss"], eval_metrics["Hit@1"], eval_metrics["Hit@10"], eval_metrics["Hit@100"])

	#     # if we are training the Category Encoder of the CAAE, so we only have Category loss 
	#     else:

	#         checkpoint_handler[criterion]["test_results"] = "Epoch {:0>2d}: | TEST | Cat_Loss: {:.4f}".format(checkpoint_handler[criterion]["epoch"], eval_metrics["Cat_Loss"])


	#     # higher BLUE means higher performance, so we are trying to minimize the value (1 - BLUE). But we need to print the original value
	#     best_value = 1 - checkpoint_handler[criterion]["best_value"] if "Hit@1" in criterion else checkpoint_handler[criterion]["best_value"]

	#     logger.info("")
	#     logger.info("  "  + criterion + "::  Best value: {:.4f}, epoch: {:0>2d}".format(best_value, checkpoint_handler[criterion]["epoch"]))

	#     logger.info("Train: " + checkpoint_handler[criterion]["train_results"])
	#     logger.info("Test : " + checkpoint_handler[criterion]["test_results"])

	#     results = args.output_dir + "\n" + checkpoint_handler[criterion]["train_results"] + "\n" + checkpoint_handler[criterion]["test_results"]

	#     # f = open("results_filename.txt", "a")
	#     # f.write(results)
	#     # f.close()



# def init(args):



#     if args.HIBERT: 

#         if args.task == "semantic":
#             Base_Model_Class = HIBERTSemanticTransformer
#             # checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss"]

#         elif args.task == "nlg":
#             Base_Model_Class = HIBERTNLGTransformer
#             # checkpoint_criteria = ["Eval_Loss", "Perplexity", "BLUE"]


#         else:

#             # checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss", "Perplexity", "BLUE"]

#             if args.task == "pretrained":
#                 Base_Model_Class = HIBERTPretrainedFullTransformer
#             else:
#                 Base_Model_Class = HIBERTJointFullTransformer

#     # Flat models
#     else:

#         if args.task == "semantic":
#             Base_Model_Class = FlatSemanticTransformer
#             # checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss"]

#         elif args.task == "nlg":
#             Base_Model_Class = FlatNLGTransformer
#             # checkpoint_criteria = ["Eval_Loss", "Perplexity", "BLUE"]


#         else:

#             # checkpoint_criteria = ["Cat_Loss", "SA_Loss", "Inter_Loss", "Perplexity", "BLUE"]

#             if args.task == "pretrained":
#                 Base_Model_Class = FlatPretrainedFullTransformer
#             else:
#                 Base_Model_Class = FlatJointFullTransformer


#     # FullAutoRec, FullCAAE, FullCatDecoder
#     # parser.add_argument("--rec_model", default="autorec", type=str) # rec_model = "autorec" or "CAAE" or "Cat2Item"

#     if args.rec_model == "autorec":
#         Model_Class = FullAutoRec
#         checkpoint_criteria = ["Loss", "Hit@1", "Hit@10", "Hit@100"]
#         base_model_name = "best_SA_Loss.pickle"
#     elif args.rec_model == "CAAE":
#         Model_Class = FullCAAE
#         if args.CAAE_encoder:
#             checkpoint_criteria = ["Cat_Loss"]
#         else:
#             checkpoint_criteria = ["Loss", "Hit@1", "Hit@10", "Hit@100"]
#         base_model_name = "best_SA_Loss.pickle"

#     else:
#         Model_Class = FullCatDecoder
#         checkpoint_criteria = ["Loss", "Hit@1", "Hit@10", "Hit@100"]
#         base_model_name = "best_Cat_Loss.pickle"




#     checkpoint_handler = {}

#     for criterion in checkpoint_criteria:
#         checkpoint_handler[criterion] = {"best_value" : 2, "epoch" : -1, "train_results" : "","test_results" : "", "filename": "best_" + criterion + ".pickle"}
#         # for Losses, we start from 1e10 and want to decrease while training (but initial best should be an unreachable uper bound)
#         if "Loss" in criterion:
#             checkpoint_handler[criterion]["best_value"] = 1e10 




#     # build base_model_dir, depending on the parameters
#     base_model_dir = "HIBERT" if args.HIBERT else "FLAT"
#     base_model_dir += "_" + args.task

#     if args.task == "semantic":
#         base_model_dir += "_" + args.CLS_mode
#         base_model_dir += "_CAT_SA_a_" + str(args.cat_sa_alpha)

#     if args.task == "full" or args.task == "pretrained":
#         base_model_dir += "_" + args.CLS_mode
#         base_model_dir += "_CAT_SA_a_" + str(args.cat_sa_alpha)
#         base_model_dir += "_SEM_NLG_a_" + str(args.sem_nlg_alpha)


#     if args.use_pretrained:
#         base_model_dir += "_Pretrained"
#     else:
#         base_model_dir += "_Hidden_" + str(args.hidden_size) + "_Layers_" + str(args.num_hidden_layers) + "_Heads_" + str(args.num_attention_heads)# + "_Inter_" + str(args.intermediate_size)

#     base_model_dir += "_BS_" + str(args.base_conversations_per_batch)

#     base_model_dir = base_model_dir if args.base_model_dir == "" else args.base_model_dir


#     output_dir = os.path.join(base_model_dir,  "Recommender")

#     output_dir += "_" + args.rec_model

#     output_dir += "_GT" if args.use_ground_truth else ""

#     output_dir += "_BS" + str(args.conversations_per_batch)

#     output_dir += "_lr_" + str(args.learning_rate)

#     output_dir += "_fine_" + str(args.finetune)


#     return checkpoint_handler, Base_Model_Class, base_model_dir, base_model_name, Model_Class, output_dir



# def update_ckeckpoint_handler(checkpoint_handler, train_metrics, eval_metrics, epoch):

#     # if we are currently training a recommending model
#     if "Hit@1" in eval_metrics:

#         train_results_text = "Epoch {:0>2d}: | TRAIN | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} | EVAL | Loss: {:.4f}, Hit@1: {:.4f}, Hit@10: {:.4f}, Hit@100: {:.4f} " \
#                     .format(epoch, train_metrics["Loss"], train_metrics["Hit@1"], train_metrics["Hit@10"], train_metrics["Hit@100"], eval_metrics["Loss"], eval_metrics["Hit@1"], eval_metrics["Hit@10"], eval_metrics["Hit@100"])
#         #  1- Hit@N, because we check if it decreases, though we actually care if it increases. (Hit@N is a ratio, and bounded in [0,1])
#         checkpoint_criteria_values = {"Loss": eval_metrics["Loss"], "Hit@1": 1 - eval_metrics["Hit@1"], "Hit@10": 1 - eval_metrics["Hit@10"], "Hit@100": 1 - eval_metrics["Hit@100"]}

#     # if we are training the Category Encoder of the CAAE, so we only have Category loss 
#     else:

#         train_results_text = "Epoch {:0>2d}: | TRAIN | Cat_Loss: {:.4f} | EVAL | Cat_Loss: {:.4f}".format(epoch, train_metrics["Cat_Loss"], eval_metrics["Cat_Loss"])

#         checkpoint_criteria_values = {"Cat_Loss": eval_metrics["Cat_Loss"]}



#     criteria_improved = []

#     for criterion in checkpoint_handler:
#         # print(epoch, " Criterion: ", criterion, ", loss: ", checkpoint_criteria_values[criterion], ", best_loss: ", checkpoint_handler[criterion]["best_value"])
#         if checkpoint_criteria_values[criterion] < checkpoint_handler[criterion]["best_value"]:
#             # reset universal patience value
#             criteria_improved.append(criterion)
#             # update best values for this criterion
#             checkpoint_handler[criterion]["best_value"] = checkpoint_criteria_values[criterion]
#             checkpoint_handler[criterion]["epoch"] = epoch
#             checkpoint_handler[criterion]["train_results"] = train_results_text



#     return checkpoint_handler, train_results_text, criteria_improved


























if __name__ == '__main__':
	main()





