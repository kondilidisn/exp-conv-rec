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


class Cat_Pref_BERT(nn.Module):
	def __init__(self, vocab_size, cat_size, n_movies, ignore_model = False, args = None):
		super(Cat_Pref_BERT, self).__init__()
		self.vocab_size = vocab_size
		self.cat_size = cat_size
		self.n_movies = n_movies
		self.args = args

		if ignore_model == False:

			self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')

			# if requested, we have to extend the parameters of the positional embeddings, so that the input length limit can be increased.
			if args.input_length_limit > 512:
				# create new positional embeddings of the appropriate new max input length
				pos_emb_weights = torch.Tensor(args.input_length_limit, 768)
				# initialize their values
				torch.nn.init.normal_(pos_emb_weights)	
				# copy first 512 positional embeddings from the pretrained bert
				pos_emb_weights [:512, :] = self.encoder.embeddings.position_embeddings.weight.clone()

				# create Embeddings, out of the parameters
				new_positional_embeddings = nn.Embedding(num_embeddings=args.input_length_limit, embedding_dim=768, _weight= pos_emb_weights)

				# replace the old positional embeddings of the model with the new ones
				self.encoder.embeddings.position_embeddings = new_positional_embeddings

				# do necessary changes in the instantiated and pretrained bert model, in order to adjust to the new max input length
				self.encoder.embeddings.register_buffer("position_ids", torch.arange(args.input_length_limit).expand((1, -1)))
				self.encoder.embeddings.position_embeddings.num_embeddings = args.input_length_limit
				self.encoder.embeddings.max_position_embeddings = args.input_length_limit
				# update parameter's value on BERT's config
				self.encoder.config.max_position_embeddings = args.input_length_limit

			# extend vocab size of the model accordingly as well
			self.encoder._resize_token_embeddings(new_num_tokens = vocab_size)

			self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=args.adam_epsilon, weight_decay=0, amsgrad=False)

			self.n_cls_tokens = self.cat_size
			# have a trainable linear function for each CLS token
			self.cat_prediction = nn.ModuleList()
			for i in range(self.n_cls_tokens):
				self.cat_prediction.append(nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1))

			self.mse_loss = torch.nn.MSELoss(reduction='mean')


	def split_batch_to_minibatches(self, batch):

		if self.args.max_samples_per_gpu == -1:
			return [batch]

		# calculate the number of minibatches so that the maximum number of samples per gpu is maintained
		size_of_minibatch = self.args.max_samples_per_gpu

		# calculate number of minibatches
		number_of_samples_in_batch = batch["contexts"].size(0)
		number_of_minibatches = number_of_samples_in_batch // size_of_minibatch
		# check if there is an incomplete last minibatch
		if number_of_samples_in_batch % size_of_minibatch != 0:
			number_of_minibatches += 1

		# arrange the minibatches
		minibatches = []
		for i in range(number_of_minibatches):
			minibatch = {}
			for key in batch:
				temp = batch[key][ i * size_of_minibatch : (i+1) * size_of_minibatch ]
				minibatch[key] = temp
			minibatches.append(minibatch)

		return minibatches


	# forward function, splits batch to minibaches, concatentes final outputs, and normalizes losses over batch, w.r.t. number of targets per task
	def forward_batch(self, batch, train = False):

		# will be used for appropriatly averaging the losses, from the minibatches, depending on the number of sampels per minibatch
		batch_loss = 0
		batch_output = []

		batch_size = batch["contexts"].size(0)

		minibatches = self.split_batch_to_minibatches(batch)

		for minibatch in minibatches:

			minibatch_size = minibatch["contexts"].size(0)

			minibatch_to_batch_ratio = minibatch_size / batch_size

			cat_pred, cat_loss = self.forward(minibatch)

			# normalize Category loses so that the represent tha minibatch w.r.t. the complete batch
			cat_loss *= minibatch_to_batch_ratio

			batch_loss += cat_loss

			batch_output.append(cat_pred)

			if cat_loss != 0 and train:
				cat_loss.backward()

		if train:
			self.optimizer.step()

		# concatenate outputs of minibatches
		batch_output = torch.cat(batch_output, dim = 0)

		return batch_output, batch_loss


	def evaluate_model(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		total_loss = 0
		num_of_samples = 0

		with torch.no_grad():

			for step in tqdm(range(n_batches)):

				# we retrieve a batch
				batch = batch_loader.load_batch(subset = subset)

				if batch == None:
					continue

				batch_size = batch["contexts"].size(0)

				num_of_samples += batch_size

				_, cat_loss = self.forward_batch(batch)

				total_loss += cat_loss.item() * batch_size

		total_loss /= num_of_samples

		return  total_loss



	def train_epoch(self, batch_loader):


		n_batches = batch_loader.n_batches["train"]

		total_loss = 0
		num_of_samples = 0

		for step in tqdm(range(n_batches)):

			self.optimizer.zero_grad()

			# we retrieve a batch
			batch = batch_loader.load_batch(subset = "train")

			if batch == None:
				continue

			batch_size = batch["contexts"].size(0)

			num_of_samples += batch_size

			_, cat_loss= self.forward_batch(batch, train=True)

			total_loss += cat_loss*batch_size

		total_loss /= num_of_samples

		return  total_loss

	def forward(self, batch):
		contexts = batch["contexts"]
		token_types = batch["token_types"]
		attention_masks = batch["attention_masks"]
		category_targets = batch["category_targets"]

		# if the model is on cuda, we transfer all tensors to cuda
		if next(self.parameters()).is_cuda:
			contexts, token_types, attention_masks, category_targets = \
			 contexts.cuda(), token_types.cuda(), attention_masks.cuda(), category_targets.cuda()

		last_hidden_state = self.encoder(input_ids = contexts, attention_mask=attention_masks, token_type_ids=token_types)[0]

		cls_input = last_hidden_state[:, : self.n_cls_tokens, :]

		# pass each CLS hidden activation, through its corresponding trainable linear function
		cat_pred = []
		for i in range(self.n_cls_tokens):
			cat_pred.append( self.cat_prediction[i]( cls_input[:,i, :] ) )
		# bring the predicted category vectors to their final form
		cat_pred = torch.stack( cat_pred, dim = 1).view(category_targets.size())

		# Use sigmoid
		cat_pred = torch.sigmoid(cat_pred)

		cat_mask = (category_targets != -1).view(category_targets.size())

		if cat_mask.sum() == 0:
			cat_loss = self.mse_loss(torch.ones(1), torch.ones(1))
			if next(self.parameters()).is_cuda:
				cat_loss = cat_loss.cuda()
		else:
			cat_loss = self.mse_loss(cat_pred[cat_mask].view(-1), category_targets[cat_mask].view(-1))

		return cat_pred, cat_loss

