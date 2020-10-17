
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import math
import datetime 
from multiprocessing.pool import ThreadPool
import os
import threading
import nltk

from tqdm import tqdm


import transformers



# we need models that load transformers and freeze it (one common parent class for all)

# the parent class again will have all evaluation functions and training, eval loops

# 1 class for autoencoder that takes the SA output, and predicts item ratings

# 1 class for CAAE that takes the SA output and predicts item ratings

# 1 class for Trans_cat -> Item decoder, that takes the CAt prediction output and translates it into item ratings.


# always evaluate once before first training epoch (epoch -1), which will be the results (pretrained-frozen)
# for each class, there will be a parameter of "recommender module loading path", which either loads the model and finetures it
#  or if empty, then initializes the recommender model and trains it.




class TextModel(nn.Module):
	def __init__(self, vocab_size, cat_size, n_movies, args = None, temperature = 0.5):
		super(TextModel, self).__init__()


		self.vocab_size = vocab_size
		self.cat_size = cat_size
		self.n_movies = n_movies
		self.args = args
		self.use_ground_truth = args.use_ground_truth

		# this will define if we are training for category prediction or for the recommendation task
		# it can only be True in the single case where we are training the CAAE Category Encoder
		self.CAAE_encoder = args.CAAE_encoder

		# loss and evaluation parameters
		self.CrossE_Loss = torch.nn.CrossEntropyLoss()
		# self.rankings = []
		self.loss_sum = 0
		self.hit_at_1_counter = 0
		self.hit_at_10_counter = 0
		self.hit_at_100_counter = 0
		self.total_number_of_targets = 0
		self.temperature = temperature


		# instantiate base model
		self.base_model = args.Base_Model_Class(vocab_size = vocab_size, cat_size = cat_size, n_movies=n_movies, args = args)


		# self = self.cuda()
		# load base model
		if args.use_ground_truth == False:
			self.base_model.load_state_dict(torch.load( os.path.join(args.base_model_dir, args.base_model_name) ))

		# if we are not finetuning, but we do transfer learning, then we freeze the parameters of the base model
		if args.finetune == False or args.use_ground_truth == True:
			for param in self.base_model.parameters():
				param.requires_grad = False



	def evaluate_model(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		if self.args.debug_run:
			n_batches = 3

		with torch.no_grad():

			# for step in range(n_batches):
			for step in tqdm(range(n_batches)):

				# we retrieve a batch
				batch = batch_loader.load_batch(subset = subset, complete = True)

				if batch == None:
					continue

				minibatches = self.base_model.split_batch_to_minibatches(batch)

				for minibatch in minibatches:
					
					if next(self.parameters()).is_cuda:
						minibatch = self.base_model.make_batch_multiple_of_GPUs_for_DataParallel(minibatch)


					output, loss = self(minibatch)

		return self.get_metrics_and_reset()


	def train_epoch(self, batch_loader):


		n_batches = batch_loader.n_batches["train"]

		if self.args.debug_run:
			n_batches = 3

		# for step in range(n_batches):
		for step in tqdm(range(n_batches)):

			self.zero_grad()

			# we retrieve a batch
			batch = batch_loader.load_batch(subset = "train", complete = True)

			if batch == None:
				continue


			batch_size = batch["contexts"].size(0)


			minibatches = self.base_model.split_batch_to_minibatches(batch)

			# we average losses over the whole batch 
			# batch_loss = torch.tensor(0).float()

			# if next(self.parameters()).is_cuda:
			#     batch_loss = batch_loss.cuda()


			for minibatch in minibatches:

				minibatch_size = minibatch["contexts"].size(0)

				minibatch_to_batch_ratio = minibatch_size / batch_size

				torch.cuda.empty_cache()

				if next(self.parameters()).is_cuda:
					minibatch = self.base_model.make_batch_multiple_of_GPUs_for_DataParallel(minibatch)
						
				torch.cuda.empty_cache()

				# do the forward pass
				output, loss = self(minibatch)

				loss *= minibatch_to_batch_ratio

				if loss != 0:
					loss.backward()


				torch.cuda.empty_cache()

				# we preprocess the losses
				if self.args.n_gpu > 1:
					loss = loss.mean()

				# add the losses of the batch
				# batch_loss += loss

			# get average losses over minibatches
			# batch_loss = batch_loss / len(minibatches)

			# depending on the task, some minibatches do not have any targets
			# if batch_loss.item() == 0:
			#     continue



			# torch.cuda.empty_cache()
			# perform backward step
			# batch_loss.backward()

			# torch.cuda.empty_cache()
			
			self.optimizer.step()

			# torch.cuda.empty_cache()


		return self.get_metrics_and_reset()


	def softmax_with_temperature(self, tensor, temperature):
		temperature_tensor = torch.ones_like(tensor)*temperature

		tempered_tensor = tensor/temperature_tensor

		output = torch.softmax(tempered_tensor, dim=-1)

		return output



	def update_hit_at_rank_metrics(self, item_rec_scores, complete_sample_movie_targets):

		_, ranked_movie_ids = torch.topk(item_rec_scores, k=100, dim=-1)
		# print(ranked_movie_ids)

		ranked_movie_ids = ranked_movie_ids.cpu().tolist()

		for i in range(complete_sample_movie_targets.size(0)):
			# get target for that sample
			target_movie_id = complete_sample_movie_targets[i].item()
			# print(target_movie_id)

			if target_movie_id == ranked_movie_ids[i][0]:
				self.hit_at_1_counter += 1
				self.hit_at_10_counter += 1
				self.hit_at_100_counter += 1

			elif target_movie_id in ranked_movie_ids[i][:10]:
				self.hit_at_10_counter += 1
				self.hit_at_100_counter += 1

			elif target_movie_id in ranked_movie_ids[i]:
				self.hit_at_100_counter += 1

	def get_metrics_and_reset(self):

		metrics = {}
		# we only normalize and return the Cat loss
		if self.CAAE_encoder:

			metrics["Cat_Loss"] = self.loss_sum / float(self.total_number_of_targets)
			self.loss_sum = 0
			self.total_number_of_targets = 0

		# we return normalized loss and ranking metrics
		else:

			metrics["Hit@1"] = self.hit_at_1_counter / float(self.total_number_of_targets)
			metrics["Hit@10"] = self.hit_at_10_counter / float(self.total_number_of_targets)
			metrics["Hit@100"] = self.hit_at_100_counter / float(self.total_number_of_targets)
			metrics["Loss"] = self.loss_sum / float(self.total_number_of_targets)

			self.hit_at_1_counter = 0
			self.hit_at_10_counter = 0
			self.hit_at_100_counter = 0
			self.loss_sum = 0
			self.total_number_of_targets = 0

		return metrics


class FullAutoRec(TextModel):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(FullAutoRec, self).__init__(vocab_size, cat_size, n_movies, args)

		self.rec_layer_sizes = args.rec_layer_sizes
		# set layer sizes
		encoder_layer_sizes = [self.n_movies] + args.rec_layer_sizes + [self.cat_size]

		decoder_layer_sizes = [self.cat_size] + list(reversed(args.rec_layer_sizes)) + [self.n_movies]

		# initialie linear layers
		self.encoder_layers = nn.ModuleList()
		self.decoder_layers = nn.ModuleList()

		for i in range( len(encoder_layer_sizes) -1):
			self.encoder_layers.append( nn.Linear(in_features=encoder_layer_sizes[i], out_features=encoder_layer_sizes[i+1]))
		for i in range( len(decoder_layer_sizes) -1):
			self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))


		# optimizer
		self.optimizer = torch.optim.Adam(self.parameters())





	def forward(self, batch):

		batch_movie_mentions, complete_sample_movie_targets = batch["batch_movie_mentions"], batch["complete_sample_movie_targets"]

		# print(complete_sample_movie_targets)
		# exit()

		base_outputs, base_losses = self.base_model(batch)


		if self.use_ground_truth:
			# cat_pred = batch["category_targets"]
			sa_pred = batch["sentiment_analysis_targets"]
		else:
			# cat_pred = base_outputs[0]
			sa_pred = base_outputs[1]


		if next(self.parameters()).is_cuda:
			sa_pred = sa_pred.cuda()
			complete_sample_movie_targets = complete_sample_movie_targets.cuda()

		sa_movie_vector = self.base_model.get_movie_ratings_vector(sa_pred, batch_movie_mentions)

		# print((sa_movie_vector != 0).sum(dim = -1)) # tensor([0, 1, 4, 5]) [We have the sentiment analysis, for each movie mentioned so far in the conv, that is followed by seeker's response]
		# print((sa_movie_vector).sum(dim = -1)) # tensor([0.0000, 0.7422, 2.4307, 3.2863])  [We have the sentiment analysis, for each movie mentioned so far in the conv, that is followed by seeker's response]

		input = sa_movie_vector

		for i in range( len(self.encoder_layers)):
			input = torch.sigmoid( self.encoder_layers[i](input))

		for i in range( len(self.decoder_layers) - 1):
			input = torch.sigmoid( self.decoder_layers[i](input))

		# the last layer uses softmax for activation function
		item_rec_scores = torch.softmax(self.decoder_layers[-1](input), dim = -1)

		# calcualte loss, on the task of predicting exactly the recommended movie
		loss = self.CrossE_Loss(item_rec_scores, complete_sample_movie_targets)
		# add loss to sum, in order to properly average the epoch loss
		total_targets = complete_sample_movie_targets.size(0)


		self.update_hit_at_rank_metrics(item_rec_scores, complete_sample_movie_targets)
		self.loss_sum += loss.item() * total_targets
		self.total_number_of_targets += total_targets



		return item_rec_scores, loss



# class TransCatPredFromSA(TextModel):
#     def __init__(self, vocab_size, cat_size, n_movies, args = None):
#         super(TransCatPredFromSA, self).__init__(vocab_size, cat_size, n_movies, args)


class FullCAAE(TextModel):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(FullCAAE, self).__init__(vocab_size, cat_size, n_movies, args)

		# args.full_caae_mode = "cat" or "item" (item can be used directly with GT Cat targets (for "hard" baseline))
		# depending on this argument, you will have different checkpoint criteria
		# if item, then you build then you use the same outputdir_with "cat", and add new 

		self.CAAE_encoder = args.CAAE_encoder


		self.rec_layer_sizes = args.rec_layer_sizes
		# set layer sizes
		encoder_layer_sizes = [self.n_movies] + args.rec_layer_sizes + [self.cat_size]

		# initialie linear layers
		self.encoder_layers = nn.ModuleList()

		for i in range( len(encoder_layer_sizes) -1):
			self.encoder_layers.append( nn.Linear(in_features=encoder_layer_sizes[i], out_features=encoder_layer_sizes[i+1], bias = False))


		# if we are currently setting up the full CAAE recommender and not only the encoder
		if self.CAAE_encoder == False:

			# load trained encoder
			self.load_state_dict(torch.load( os.path.join(args.output_dir, "best_Cat_Loss.pickle") ))
			# freeze its parameters
			for param in self.encoder_layers.parameters():
				param.requires_grad = False


			# initialize decoder
			decoder_layer_sizes = [self.cat_size] + list(reversed(args.rec_layer_sizes)) + [self.n_movies]

			self.decoder_layers = nn.ModuleList()

			for i in range( len(decoder_layer_sizes) -1):
				self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))


		# optimizer
		self.optimizer = torch.optim.Adam(self.parameters())


		self.cat_mse_loss = nn.MSELoss(reduction='sum')



	def forward(self, batch):

		batch_movie_mentions, complete_sample_movie_targets, category_targets = batch["batch_movie_mentions"], batch["complete_sample_movie_targets"], batch["category_targets"]




		if self.use_ground_truth:
			# cat_pred = batch["category_targets"]
			sa_pred = batch["sentiment_analysis_targets"]
		else:
			# cat_pred = base_outputs[0]
			base_outputs, base_losses = self.base_model(batch)
			sa_pred = base_outputs[1]

		# Category Encoder pass
		if next(self.parameters()).is_cuda:
			sa_pred = sa_pred.cuda()
			category_targets = category_targets.cuda()
			complete_sample_movie_targets = complete_sample_movie_targets.cuda()

		sa_movie_vector = self.base_model.get_movie_ratings_vector(sa_pred, batch_movie_mentions)

		# print((sa_movie_vector != 0).sum(dim = -1)) # tensor([0, 1, 4, 5]) [We have the sentiment analysis, for each movie mentioned so far in the conv, that is followed by seeker's response]
		# print((sa_movie_vector).sum(dim = -1)) # tensor([0.0000, 0.7422, 2.4307, 3.2863])  [We have the sentiment analysis, for each movie mentioned so far in the conv, that is followed by seeker's response]

		input = sa_movie_vector

		for i in range( len(self.encoder_layers) -1 ):
			input = torch.sigmoid( self.encoder_layers[i](input))



		# # the last layer uses softmax for activation function (for predicting user category preference distribution)
		# cat_pred_scores = torch.softmax(self.encoder_layers[-1](input), dim = -1)



		cat_pred_scores = self.encoder_layers[-1](input)

		# cat_pred_scores = torch.sigmoid(cat_pred_scores)


		# cat_pred_scores = self.softmax_with_temperature(cat_pred_scores, self.temperature)

		cat_pred_scores = torch.sigmoid(cat_pred_scores)

		# # USE SUM NORMALIZATION INSTEAD !
		# cat_pred_scores = torch.nn.functional.relu(self.encoder_layers[-1](input))
		# # print(category_scores)
		# # normalize by sum
		# sums = cat_pred_scores.sum(dim = -1)
		# mask = sums == 0
		# sums[mask] += 1
		# cat_pred_scores = cat_pred_scores / sums.unsqueeze(-1).repeat(1,cat_pred_scores.size(-1))

		Cat_Loss = self.cat_mse_loss(cat_pred_scores, category_targets)

		# add loss to sum, in order to properly average the epoch loss
		total_targets = complete_sample_movie_targets.size(0)

		# If we are currently training the Category Ecnoder
		if self.CAAE_encoder:

			self.loss_sum += Cat_Loss.item() * total_targets
			self.total_number_of_targets += total_targets

			return cat_pred_scores, Cat_Loss

		# if we are using the complete CAAE model
		else:

			input = cat_pred_scores

			for i in range( len(self.decoder_layers) - 1):
				input = torch.sigmoid( self.decoder_layers[i](input))


			output = self.decoder_layers[-1](input)

			# the last layer uses softmax for activation function
			item_rec_scores = torch.softmax(output, dim = -1)

			# calcualte loss, on the task of predicting exactly the recommended movie
			loss = self.CrossE_Loss(output, complete_sample_movie_targets)


			self.update_hit_at_rank_metrics(item_rec_scores, complete_sample_movie_targets)
			self.loss_sum += loss.item() * total_targets
			self.total_number_of_targets += total_targets

			return item_rec_scores, loss


class FullCatDecoder(TextModel):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(FullCatDecoder, self).__init__(vocab_size, cat_size, n_movies, args)


		self.rec_layer_sizes = args.rec_layer_sizes

		# set layer sizes
		decoder_layer_sizes = [self.cat_size] + list(reversed(args.rec_layer_sizes)) + [self.n_movies]

		# initialie linear layers
		self.decoder_layers = nn.ModuleList()

		for i in range( len(decoder_layer_sizes) -1):
			self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))


		# optimizer
		self.optimizer = torch.optim.Adam(self.parameters())





	def forward(self, batch):

		batch_movie_mentions, complete_sample_movie_targets = batch["batch_movie_mentions"], batch["complete_sample_movie_targets"]

		# print(complete_sample_movie_targets)
		# exit()



		if self.use_ground_truth:
			cat_pred = batch["category_targets"]
			# sa_pred = batch["sentiment_analysis_targets"]
		else:
			# print(batch["contexts"])
			base_outputs, base_losses = self.base_model(batch)
			cat_pred = base_outputs[0]
			# print("Predicted:", cat_pred)
			# print ("GT:", batch["category_targets"])
			# print("Difference :",  cat_pred - batch["category_targets"])
			# sa_pred = base_outputs[1]



		input = cat_pred

		if next(self.parameters()).is_cuda:
			input = input.cuda()
			complete_sample_movie_targets = complete_sample_movie_targets.cuda()

		for i in range( len(self.decoder_layers) - 1):
			input = torch.sigmoid( self.decoder_layers[i](input))


		output = self.decoder_layers[-1](input)

		# the last layer uses softmax for activation function
		item_rec_scores = torch.softmax(output, dim = -1)

		# calcualte loss, on the task of predicting exactly the recommended movie
		loss = self.CrossE_Loss(output, complete_sample_movie_targets)
		# add loss to sum, in order to properly average the epoch loss
		total_targets = complete_sample_movie_targets.size(0)


		self.update_hit_at_rank_metrics(item_rec_scores, complete_sample_movie_targets)
		self.loss_sum += loss.item() * total_targets
		self.total_number_of_targets += total_targets



		return item_rec_scores, loss

