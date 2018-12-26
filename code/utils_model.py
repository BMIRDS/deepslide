# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Using a ResNet to train and test.

import config
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import copy
import time
import random
from sklearn.metrics import confusion_matrix
import operator

###########################################
############# MISC FUNCTIONS ##############
###########################################

#computing the confusion matrix
def update_confusion_matrix(all_labels, all_predicts, batch_labels, batch_predicts, num_classes):

    if all_labels is not None:
        assert all_labels.shape[0] == all_predicts.shape[0]
        if all_labels.shape[0] > 10000:
            all_labels = all_labels[-10000:]
            all_predicts = all_predicts[-10000:]

    if all_labels is None and all_predicts is None:
        all_labels = batch_labels
        all_predicts = batch_predicts
    elif all_labels is not None and all_predicts is not None:
        all_labels = torch.cat((all_labels, batch_labels))
        all_predicts = torch.cat((all_predicts, batch_predicts))

    conf_matrix = confusion_matrix(all_labels, all_predicts, labels=list(range(num_classes)))

    probs_matrix = np.zeros(conf_matrix.shape)

    for i in range(probs_matrix.shape[0]):
        row = conf_matrix[i]
        if np.sum(row) == 0:
            probs_row = 0
        else:
            probs_row = row/np.sum(row)
        probs_matrix[i] = probs_row

    probs_matrix = np.around(probs_matrix, decimals=5)
    return probs_matrix, all_labels, all_predicts

#printing the confusion matrix during training
def print_conf_matrix(confusion_matrix, classes):
    first_line = "   " + " ".join(['{:5s}'.format(_class) for _class in classes])
    print(first_line)
    for row, _class in zip(confusion_matrix, classes):
        row_pretty = '{:3s}'.format(_class) + " ".join(['{:.3f}'.format(number) for number in row])
        print(row_pretty)

#random rotation function
#credits to Naofumi Tomita
class Random90Rotation():
    def __init__(self, degrees=[0, 90, 180, 270]):
        self.degrees = degrees

    def __call__(self, im):
        degree = random.sample(self.degrees, k=1)[0]
        return im.rotate(degree)

#instantiate the model
def create_model(num_layers, pretrain):

    assert num_layers in [18, 24, 50, 101, 152]
    architecture = 'resnet' + str(num_layers)
    model = None

    #for pretrained on imagenet
    if pretrain == True:
        if architecture == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif architecture == 'resnet34':
            model = torchvision.models.resnet34(pretrained=True)
        elif architecture == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif architecture == 'resnet101':
            model = torchvision.models.resnet101(pretrained=True)
        elif architecture == 'resnet152':
            model = torchvision.models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.num_classes)

    #default he initialization
    else:
        if architecture == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=config.num_classes)
        elif architecture == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False, num_classes=config.num_classes)
        elif architecture == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=config.num_classes)
        elif architecture == 'resnet101':
            model = torchvision.models.resnet101(pretrained=False, num_classes=config.num_classes)
        elif architecture == 'resnet152':
            model = torchvision.models.resnet152(pretrained=False, num_classes=config.num_classes)
        
    return model

#get the data transforms:
def get_data_transforms():
	
	data_transforms = {
	    'train': transforms.Compose([
	        transforms.CenterCrop(config.patch_size),
	        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
	        transforms.RandomHorizontalFlip(),
	        transforms.RandomVerticalFlip(),
	        Random90Rotation(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15]) #mean and standard deviations for lung adenocarcinoma resection slides
	    ]),
	    'val': transforms.Compose([
	        transforms.CenterCrop(config.patch_size),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15])
	    ]),
	    'unnormalize': transforms.Compose([
	        transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
	    ]),
	}

	return data_transforms

#printing the model
def print_params(train_folder, num_epochs, num_layers, 
				learning_rate, batch_size, 
				weight_decay, learning_rate_decay, 
				resume_checkpoint, resume_checkpoint_path, 
				save_interval, checkpoints_folder, 
				pretrain, log_csv):

	print("train_folder:", train_folder)
	print("num_epochs:", num_epochs)
	print("num_layers:", num_layers)
	print("learning_rate:", learning_rate)
	print("batch_size:", batch_size)
	print("weight_decay:", weight_decay)
	print("resume_checkpoint:", resume_checkpoint)
	print("resume_checkpoint_path (only if resume_checkpoint is true):", resume_checkpoint_path)
	print("save_interval:", save_interval)
	print("output in checkpoints_folder:", checkpoints_folder)
	print("pretrain:", pretrain)
	print("log_csv:", log_csv)
	print()


###########################################
########## MAIN TRAIN FUNCTION ############
###########################################

#helper function for training resnet
def train_helper(model, dataloaders, device, dataset_sizes, criterion, optimizer, scheduler, num_epochs, save_interval, writer):

    since = time.time()

    #each epoch
    for epoch in range(num_epochs):

        ############### training phase ###############
        phase = 'train'
        model.train()

        train_running_loss = 0.0
        train_running_corrects = 0
        train_conf_matrix = None
        train_all_labels = None
        train_all_predicts = None

        #train over all training data
        for inputs, labels in dataloaders['train']:
            train_inputs = inputs.to(device)
            train_labels = labels.to(device)
            optimizer.zero_grad()

            #forward and backprop
            with torch.set_grad_enabled(phase == 'train'):
                train_outputs = model(train_inputs)
                _, train_preds = torch.max(train_outputs, 1)
                train_loss = criterion(train_outputs, train_labels)
                train_loss.backward()
                optimizer.step()
                optimizer.param_groups

            #update training diagnostics
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(train_preds == train_labels.data)
            train_conf_matrix, train_all_labels, train_all_predicts = update_confusion_matrix(train_all_labels, train_all_predicts, train_labels.data, train_preds, config.num_classes)

        #print training diagnostics
        train_loss = train_running_loss / dataset_sizes['train']
        train_acc = train_running_corrects.double() / dataset_sizes['train']
        print("training confusion matrix:")
        print_conf_matrix(train_conf_matrix, config.classes)

        ############### validation phase ###############
        phase = 'val'
        model.eval()

        val_running_loss = 0.0
        val_running_corrects = 0
        val_conf_matrix = None
        val_all_labels = None
        val_all_predicts = None

        #forward prop over all validation data
        for val_inputs, val_labels in dataloaders['val']:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            #forward
            with torch.set_grad_enabled(phase == 'val'):
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_loss = criterion(val_outputs, val_labels)

            #update validation diagnostics
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data)
            val_conf_matrix, val_all_labels, val_all_predicts = update_confusion_matrix(val_all_labels, val_all_predicts, val_labels.data, val_preds, config.num_classes)

        #print validation diagnostics
        val_loss = val_running_loss / dataset_sizes['val']
        val_acc = val_running_corrects.double() / dataset_sizes['val']
        print("validation confusion matrix:")
        print_conf_matrix(val_conf_matrix, config.classes)

        #scheduler.step(val_loss) if you use decay on plateau
        scheduler.step()
        current_lr = None
        for group in optimizer.param_groups:
            current_lr = group['lr']

        #remaining things related to training
        if epoch % int(save_interval) == 0:
            epoch_output_path = config.checkpoints_folder + "/resnet" + str(config.num_layers) + "_e" + str(epoch) + "_va" + str(float(val_acc))[:5] + ".pt"
            confirm_output_folder(basefolder(epoch_output_path))
            torch.save(model, epoch_output_path)

        writer.write('{},{:4f},{:4f},{:4f},{:4f}\n'.format(str(epoch), train_loss, train_acc, val_loss, val_acc))

		#remaining diagnostics
        print('Epoch {} with lr {:.15f}: t_loss: {:.4f} t_acc: {:.4f} v_loss:{:.4f} v_acc: {:.4f}'.format(str(epoch), current_lr, train_loss, train_acc, val_loss, val_acc))
        print()

    # at the end:
    print()
    time_elapsed = time.time() - since
    print('training complete in {:.0f} minutes'.format(time_elapsed // 60))

#main function for training resnet
def train_resnet(train_folder, num_epochs, num_layers, 
				learning_rate, batch_size, 
				weight_decay, learning_rate_decay, 
				resume_checkpoint, resume_checkpoint_path, 
				save_interval, checkpoints_folder, 
				pretrain, log_csv):

	#loading in the data
	data_transforms = {
	    'train': transforms.Compose([
	        transforms.CenterCrop(config.patch_size),
	        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
	        transforms.RandomHorizontalFlip(),
	        transforms.RandomVerticalFlip(),
	        Random90Rotation(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15]) #mean and standard deviations for lung adenocarcinoma resection slides
	    ]),
	    'val': transforms.Compose([
	        transforms.CenterCrop(config.patch_size),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15])
	    ]),
	    'unnormalize': transforms.Compose([
	        transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
	    ]),
	}

	image_datasets = {x: datasets.ImageFolder(os.path.join(config.train_folder, x), data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=x=='train', num_workers=4) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	print(len(config.classes), 'classes:', config.classes)
	print('num train images', len(dataloaders['train'])*batch_size)
	print('num val images', len(dataloaders['val'])*batch_size)
	print("CUDA is_available:", torch.cuda.is_available())
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	#initialize model
	if resume_checkpoint == True:
		model = torch.load(resume_checkpoint_path)
		print('model loaded from', resume_checkpoint_path)
	else:
		model = create_model(num_layers, pretrain)

	model = model.to(device) #same as model.cuda()

	#multi class cross entropy
	criterion = nn.CrossEntropyLoss()

	#adam optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	#learning rate: exponential, can also try scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
	scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

	#logging the model after every epoch
	confirm_output_folder(basefolder(log_csv))
	writer = open(log_csv, 'w')
	writer.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

	#print
	print_params(train_folder, num_epochs, num_layers, 
				learning_rate, batch_size, 
				weight_decay, learning_rate_decay, 
				resume_checkpoint, resume_checkpoint_path, 
				save_interval, checkpoints_folder, 
				pretrain, log_csv)


	#train model
	model = train_helper(model, dataloaders, device, dataset_sizes, criterion, optimizer, scheduler, num_epochs, save_interval, writer)



###########################################
###### MAIN EVALUATION FUNCTION ###########
###########################################

#parsing the validation accuracy from filename
def parse_val_acc(model_path):
	no_extension = ".".join(basename(model_path).split('.')[:-1])
	val_acc = float(no_extension.split('_')[-1][2:])
	return val_acc

#return the model with the best validation accuracy
def get_best_model(checkpoints_folder):
	models = get_image_paths(checkpoints_folder)
	model_to_val_acc = {model: parse_val_acc(model) for model in models}
	best_model = max(model_to_val_acc.items(), key=operator.itemgetter(1))[0]
	return best_model

#main function for running on all the generated windows
def get_predictions(patches_eval_folder, auto_select, eval_model, checkpoints_folder, output_folder):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

	#initialize the model
	model = None
	if auto_select:
		model_path = get_best_model(checkpoints_folder)
	else:
		model_path = eval_model

	model = torch.load(model_path)
	model.train(False) 
	print('model loaded from', model_path)

	#for outputting the predictions
	class_num_to_class = {i:config.classes[i] for i in range(len(config.classes))}
	class_to_class_num = {v: k for k, v in class_num_to_class.items()}

	#data transforms, no augmentation this time.
	data_transforms = {
		'normalize': transforms.Compose([
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15])
	    ]),
	    'unnormalize': transforms.Compose([
	        transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
	    ]),
	}

	start = time.time()

	#load data for each folder:
	image_folders = get_subfolder_paths(patches_eval_folder) 
	
	for image_folder in image_folders: #for each whole slide

		#where we want to write out the predictions
		confirm_output_folder(output_folder)
		csv_path = join(output_folder, image_folder.split('/')[-1])+'.csv'
		writer = open(csv_path, 'w')
		writer.write("x,y,prediction,confidence\n")

		#load the image dataset
		image_dataset = datasets.ImageFolder(image_folder, data_transforms['normalize'])
		dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
		num_test_image_windows = len(dataloader)*config.batch_size

		#load the image names so we know the coordinates of the windows we are predicting
		image_folder = join(image_folder, image_folder.split('/')[-1])
		window_names = get_image_paths(image_folder) 

		print("testing on", num_test_image_windows, 'crops from', image_folder)
		batch_num = 0

		#loop through all the windows
		for test_inputs, test_labels in dataloader:

			batch_window_names = window_names[batch_num*config.batch_size:batch_num*config.batch_size+config.batch_size]
			test_inputs = test_inputs.to(device)
			test_outputs = model(test_inputs)
			softmax_test_outputs = nn.Softmax()(test_outputs)
			confidences, test_preds = torch.max(softmax_test_outputs, 1)

			for i in range(test_preds.shape[0]):
				#unnormalized_image = data_transforms['unnormalize'](test_inputs[i]).cpu().numpy()
				#unnormalized_image = np.swapaxes(unnormalized_image, 0, 2) #in case you want to make sure you're looking at the right image

				#get coordinates and predicted class
				image_name = batch_window_names[i]
				x = basename(image_name).split('.')[0].split(';')[0]
				y = basename(image_name).split('.')[0].split(';')[1]
				predicted_class = class_num_to_class[test_preds[i].data.item()]
				confidence = confidences[i].data.item()

				#write them 
				out_line = ','.join([x, y, predicted_class, str(confidence)[:5]]) + '\n'
				writer.write(out_line)

				#out_path = 'check_images_2/' + image_folder.split('/')[-1] + '_' + str(counter) + '_' + '_'.join([x, y, predicted_class, str(confidence)[:5]]) + '.jpg'
				#imsave(out_path, unnormalized_image)

			batch_num += 1

		writer.close()

	print('time for', patches_eval_folder, ':', time.time()-start, 'seconds')

















