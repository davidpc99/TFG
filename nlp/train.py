"""Train the model"""

import argparse
import logging
import os

from transformers.optimization import get_scheduler

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def send_metrics_to_txt(train_accuracy, validation_accuracy, train_f1, validation_f1, scheduler_learning_rate, model_dir):
    train_accuracy = list(map(str, train_accuracy))
    validation_accuracy = list(map(str, validation_accuracy))
    train_f1 = list(map(str, train_f1))
    validation_f1 = list(map(str, validation_f1))
    scheduler_learning_rate = list(map(str, scheduler_learning_rate
                                       ))
    with open(model_dir+'/train_accuracy.txt', 'w+') as f:
        for accuracy in train_accuracy:
            f.write(accuracy + '\n')
            
    with open(model_dir+'/validation_accuracy.txt', 'w+') as f:
        for accuracy in validation_accuracy:
            f.write(accuracy + '\n')
            
    with open(model_dir+'/train_f1.txt', 'w+') as f:
        for f1 in train_f1:
            f.write(f1 + '\n')
            
    with open(model_dir+'/validation_f1.txt', 'w+') as f:
        for f1 in validation_f1:
            f.write(f1 + '\n')
                
    with open(model_dir+'/scheduler.txt', 'w+') as f:
        for lr in scheduler_learning_rate:
            f.write(lr + '\n')
            

def train(model, optimizer, lr_scheduler, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        lr_scheduler.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    return metrics_mean


def train_and_evaluate(model, train_data, val_data, optimizer, lr_scheduler, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    train_accuracy = []
    validation_accuracy = []
    train_f1 = []
    validation_f1 = []
    scheduler_learning_rate = []
    best_val_acc = 0.0
    
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)
        train_metrics = train(model, optimizer, lr_scheduler, loss_fn, train_data_iterator,
              metrics, params, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)
        val_metrics, summ = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        
        scheduler_learning_rate.append(lr_scheduler.get_last_lr()[0])
        train_accuracy.append(train_metrics['accuracy'])
        validation_accuracy.append(val_metrics['accuracy'])
        train_f1.append(train_metrics['f1'])
        validation_f1.append(val_metrics['f1'])

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        
    #epoch_list = [item for item in range(1, params.num_epochs+1)]
    #xaxis = np.array(epoch_list)
    #yaxis = np.array(scheduler_learning_rate)
    #    
    #plt.figure("Scheduler")
    #plt.title("Scheduler learning rate")
    #plt.xlabel("Epoch")
    #plt.ylabel('Learning Rate')
    #plt.plot(xaxis, yaxis)
    #    
    #plt.figure("Train")
    #plt.title("Train accuracy")
    #plt.xlabel("Epoch")
    #plt.ylabel('Accuracy')
    #plt.plot(xaxis, np.array(train_accuracy))
    #plt.show()
    #    
    #plt.figure("Validation")
    #plt.title("Validation accuracy")
    #plt.xlabel("Epoch")
    #plt.ylabel('Accuracy')
    #plt.plot(xaxis, np.array(validation_accuracy))
    #plt.show()
    #    
    #plt.figure("Train & Validation")
    #plt.title("Model accuracy")
    #plt.xlabel("Epoch")
    #plt.ylabel('Accuracy')
    #plt.plot(xaxis, np.array(train_accuracy), label='Train')
    #plt.plot(xaxis, np.array(validation_accuracy), label='Validation')
    #plt.legend()
    #plt.show()
    
    return train_accuracy, validation_accuracy, train_f1, validation_f1, scheduler_learning_rate


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    params.pad_ind = data_loader.pad_ind

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    #lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=50, num_training_steps=400)
    num_batches = params.train_size//params.batch_size
    num_steps = num_batches*params.num_epochs
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=num_steps/10, num_training_steps=num_steps)
    #scheduler.get_last_lr()[0]

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_accuracy, validation_accuracy, train_f1, validation_f1, scheduler_learning_rate = train_and_evaluate(model, train_data, val_data, optimizer, lr_scheduler, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
    send_metrics_to_txt(train_accuracy, validation_accuracy, train_f1, validation_f1, scheduler_learning_rate, args.model_dir)
