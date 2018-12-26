# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Training the resnet

import utils_model
from utils_model import *


if __name__ == '__main__':

	train_resnet(	train_folder = config.train_folder,
					num_epochs = config.num_epochs,
					num_layers = config.num_layers,
					learning_rate = config.learning_rate,
					batch_size = config.batch_size,
					weight_decay = config.weight_decay,
					learning_rate_decay = config.learning_rate_decay,
					resume_checkpoint = config.resume_checkpoint,
					resume_checkpoint_path = config.resume_checkpoint_path,
					save_interval = config.save_interval,
					checkpoints_folder = config.checkpoints_folder,
					pretrain = config.pretrain, 
					log_csv = config.log_csv
		)
