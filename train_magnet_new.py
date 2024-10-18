import argparse
import os
import sys
import time
import torch

import numpy                    as np
import torchvision.models       as models
import torchvision.transforms   as transforms
import torchvision.datasets     as datasets


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from datasets                   import ImageNet , magnet_CIFAR10
from visualizer.visualizer import VisdomLinePlotter

from models                     import magnetInception
from tensorboardX               import SummaryWriter    as Logger
from util.torch_utils           import to_var, save_checkpoint, AverageMeter, accuracy
from util                       import magnet_loss, triplet_loss, softkNN_metrics, softkNC_metrics
from torch.optim.lr_scheduler   import MultiStepLR
from IPython                    import embed
from sklearn.cluster            import KMeans
from torch.utils.data.sampler   import Sampler


import torch.nn as nn
import torch.nn.functional as F

############  get_loaders
def get_loaders():

    # Data loading code (From PyTorch example https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print("Generating Datasets")

    if args.dataset == "cifar10":
        print('==> Preparing data..')

        transform_train = train_transform

        transform_test = valid_transform

        
        # train_dataset = CIFAR10(root='../data', train=True, transform=transform_train, download=True)
        # valid_dataset = CIFAR10(root='../data', train=False, transform=transform_test, download=True)
            
        train_dataset = magnet_CIFAR10(root='./data_cifar10', train=True, transform=transform_train, download=False)
        valid_dataset = magnet_CIFAR10(root='./data_cifar10', train=False, transform=transform_test, download=False)
    
    else:
        print("No dataset generated!")
        exit()
    # END IF

    print("Generating Data Loaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last = False
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return train_loader, valid_loader


###############   define_order

def define_order(cluster_assignment, cluster_centers, loss_vector):
    # Convert loss_vector to a NumPy array if it's not already
    loss_vector = np.array(loss_vector)
    print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT args.num_classes = ", args.num_classes)
    num_clusters = args.num_classes * args.K
    cluster_distances = np.ones((num_clusters, num_clusters))

    # Calculate distances between cluster centers
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            d = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            cluster_distances[i][j] = d
            cluster_distances[j][i] = d

    # Construct useful dict
    cluster_dict = {i: [] for i in range(num_clusters)}

    for i in range(len(cluster_assignment)):
        cluster_dict[cluster_assignment[i]].append(i)

    cluster_to_remove = []
    for i in range(num_clusters):
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT num_clusters = ", num_clusters)
        if len(cluster_dict[i]) < 4:
            loss_vector[i] = 0.0
            cluster_to_remove.append(i)

    # Normalize loss_vector safely
    total_loss = np.sum(loss_vector)

    if total_loss <= 0:
        print("Total loss is zero or negative; setting loss_vector to uniform distribution.")
        loss_vector = np.ones(num_clusters) / num_clusters
    else:
        # Ensure loss_vector is non-negative before normalization
        loss_vector[loss_vector < 0] = 0  # This line will now work correctly
        loss_vector /= total_loss

    # Check for NaNs after normalization and replace them
    if np.any(np.isnan(loss_vector)):
        print("Loss vector contains NaNs after normalization:", loss_vector)
        loss_vector = np.nan_to_num(loss_vector, nan=0.0)  # Replace NaNs with 0

    NONZERO_CLUSTERS = num_clusters - len(cluster_to_remove)
    print("Number of proper clusters: " + str(NONZERO_CLUSTERS))

    order = []
    
    # Construct batches
    for B in range(args.NUM_BATCHES):
        M0 = np.random.choice(range(num_clusters), p=loss_vector)

        C0K0 = (M0 // args.K) * args.K  # First cluster index in that class
        bad_clusters = cluster_to_remove + list(range(C0K0, C0K0 + args.K))

        near_M = np.argsort(cluster_distances[M0])
        near_M = [m for m in near_M if m not in bad_clusters]  # Imposter clusters in order of closeness

        near_imposter = near_M[0:args.M - 1]
        near_imposter.append(M0)

        if len(near_imposter) < args.M:
            continue

        ignore_batch = False
        for Mi in near_imposter:
            if len(cluster_dict[Mi]) < args.D:
                ignore_batch = True

        if ignore_batch:
            continue

        for Mi in near_imposter:
            samplesMi = np.random.permutation(len(cluster_dict[Mi]))[0:args.D]
            my_s = []
            for s in samplesMi:
                order.append(cluster_dict[Mi][s])
                my_s.append(cluster_dict[Mi][s])

            for s in my_s:
                cluster_dict[Mi].remove(s)

            if len(cluster_dict[Mi]) < args.D:
                loss_vector[Mi] = 0.0
                # Re-normalize loss_vector after modification
                total_loss = np.sum(loss_vector)
                
                if total_loss <= 0:
                    print("Total loss is zero or negative after modification; setting loss_vector to uniform distribution.")
                    loss_vector = np.ones(num_clusters) / num_clusters
                else:
                    loss_vector[loss_vector < 0] = 0  # Ensure non-negative before normalization
                    loss_vector /= total_loss

                # Check for NaNs after re-normalization and replace them
                if np.any(np.isnan(loss_vector)):
                    print("Loss vector contains NaNs after re-normalization:", loss_vector)
                    loss_vector = np.nan_to_num(loss_vector, nan=0.0)  # Replace NaNs with 0

    # return order
    print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT order = ", order)
    return order

################# cluster_assignment
def cluster_assignment(indices, embeddings, labels, init_centers):
    cluster_nums = args.num_classes * args.K

    print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT len(embeddings) = ", len(embeddings))
    print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT len(labels) = ", len(labels))

    print("cluster_assignment :", cluster_nums)
    assignment = [-1] * len(indices)
    cluster_centers = np.zeros((cluster_nums, args.embedding_size))
    class_dict = {}

    # Create a class dict and initialize it with all the embeddings
    if args.experiment_type == "exp1":
        classs_to_process = range(args.num_classes)
    else:
        classs_to_process =  args.current_classes
        
    for i in classs_to_process:
        print("Processing class:", i)
        class_dict[i] = {'embeddings': [], 'indices': []}
        
        for j in range(len(labels)):
            if labels[j] == i:
                class_dict[i]['embeddings'].append(embeddings[j])
                class_dict[i]['indices'].append(indices[j])

        # Convert list to a single array
        class_dict[i]['embeddings'] = np.asarray(class_dict[i]['embeddings'])
        
        # Skip empty classes
        if class_dict[i]['embeddings'].shape[0] == 0:
            print(f"Warning: Class {i} has no embeddings. Skipping.")
            continue

        # Calculate K-Means++ clustering
        c_init_centers = init_centers[i * args.K:(i + 1) * args.K, :] if init_centers is not None else 'k-means++'

        kmeans = KMeans(n_clusters=args.K, init=c_init_centers).fit(class_dict[i]['embeddings'])
        k_labels = kmeans.labels_
        k_centers = kmeans.cluster_centers_

        for k in range(args.K):
            global_k = (i * args.K) + k
            cluster_centers[global_k] = k_centers[k]

            # Assign each embedding to its corresponding cluster
            for l in range(len(k_labels)):
                if k_labels[l] == k:
                    assignment[class_dict[i]['indices'][l]] = global_k

    # Ensure all indices are assigned by checking for unassigned indices
    for idx in range(len(assignment)):
        if assignment[idx] == -1:
            # Assign to the nearest cluster center
            nearest_cluster = np.argmin(np.linalg.norm(cluster_centers - embeddings[idx], axis=1))
            assignment[idx] = nearest_cluster
            print(f"Index {idx} was unassigned. Assigned to nearest cluster {nearest_cluster}.")

    assert not (-1 in assignment), "Error: Not all indices were assigned!"
    
    print("cluster_assignment : Done")
    return cluster_centers, assignment



################# indexing_step

def indexing_step(model, data_loader, cluster_centers):

        model.eval()
        curr_time = time.time()

        t_embeddings  = []
        t_labels      = []
        t_indices     = []

        data_loader.dataset.default_read_order()

        for i, (input, target, indices) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input, requires_grad=True)
            # compute output
            output = model(input_var)

            t_embeddings += list(output.cpu().data.numpy())
            t_labels     += list(target.numpy())
            t_indices    += list(indices.numpy())
            del input, target, indices

        # End FOR

        print('KNN: Forward Pass Time   {time:.3f}'.format(time=time.time() - curr_time))
        curr_time = time.time()
        cluster_centers, assignment = cluster_assignment(t_indices, t_embeddings, t_labels, init_centers= cluster_centers)
        print('KNN: Clustering time     {time:.3f}'.format(time=time.time() - curr_time))

        return cluster_centers, assignment


########################   eval_step
def eval_step( model, data_loader,  criterion, step, datasplit, stdev, cluster_centers):
    batch_time = AverageMeter()
    losses = AverageMeter()

    metrics = softkNC_metrics(stdev = stdev, cluster_centers = cluster_centers ,K = args.K, L = args.L)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, inst_indices) in enumerate(data_loader):
        input_var = torch.autograd.Variable(input.cuda())
        # target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        target_var = target.cuda(non_blocking=True)


        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        metrics.update(output, target)

    print("Evaluation: Forward Embedding Calculation (Time Elapsed {time:.3f})".format(time=time.time() - end))
    curr_time = time.time()
    acc1, acc5 = metrics.accuracy(stdev = stdev)

    print('Test: \t'
          'Time {test_time:.3f}\t'
          # 'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1:.3f}\t'
          'Prec@5 {top5:.3f}'.format(
           i, len(data_loader), test_time=time.time() - curr_time, loss=losses
           , top1=acc1, top5=acc5)
    )

    # # logger.add_scalar(args.dataset + "/Loss valid  (avg)",   losses.avg, step)
    # logger.add_scalar(args.dataset + "/Perc5 " + datasplit + " (avg)",   acc5,   step)
    # logger.add_scalar(args.dataset + "/Perc1 " + datasplit + " (avg)",   acc1,   step)

    return losses.avg, 0.0



################################# train_step

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None, assignment = None, loss_vector = None, loss_count=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1    = AverageMeter()
    top5    = AverageMeter()
    stdev    = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, inst_indices) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        loss, c_loss_vector, c_loss_count, c_stdev = criterion(output, list(inst_indices.numpy()), assignment, loss_vector, loss_count, input, model)
        stdev.update(c_stdev, 1)

        loss_vector *= 0.9
        loss_count  *= 0.9
        loss_vector += (c_loss_vector/100.)
        loss_count  += c_loss_count

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        

        # compute gradient and do SGD step
        model.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        del loss, output
        batch_time.update(time.time() - end)
        
        if i % args.log_rate == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


            # Log items
            logger.add_scalar("Misc/batch time (s)",    batch_time.avg,                                   step + i)
            logger.add_scalar("Misc/Train_%",           1.0 - (data_time.avg/batch_time.avg),             step + i)
            logger.add_scalar("Misc/epoch time (min)",  batch_time.sum / 60.,                             step + i)
            logger.add_scalar("Misc/time left (min)",   ((len(train_loader) - i) * batch_time.avg) / 60., step + i)
            logger.add_scalar(args.dataset + "/Loss train  (avg)",          losses.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc5 train (avg)",          top5.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc1 train (avg)",          top1.avg,                      step + i)

        end = time.time()

        if valid_loader != None and i % args.eval_step == 0 and i > 0:
            _, _ = eval_step(   model       = model,
                                data_loader = valid_loader,
                                criterion   = criterion,
                                step        = step + i,
                                datasplit   = "valid")
            model.train()
            

    # return loss_vector, loss_count, stdev.avg
    return loss_vector, loss_count, torch.tensor(stdev.avg)

################################   Main function


def main(args):
    curr_time = time.time()
    
    global plotter
    plotter = VisdomLinePlotter(env_name=args.name)

    print("#############  Read in Database   ##############")
    train_loader, valid_loader = get_loaders()                    
    
    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("######## Initiate Model and Optimizer   ##############")  
    model   = magnetInception(args.embedding_size)

    
    if args.resume is not None:
        print("Loading pretrained Module")
        checkpoint      = torch.load(args.resume)
        state_dict      = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    model   = torch.nn.DataParallel(model).cuda()


    # Criterion was not specified by the paper, it was assumed to be cross entropy (as commonly used)
    if args.loss == "magnet":
        criterion = magnet_loss(D = args.D, M = args.M, alpha = args.GAP).cuda()    # Loss function
    elif args.loss == "triplet":
        criterion = triplet_loss(alpha = 0.304).cuda()    # Loss function
    elif args.loss == "softmax":
        criterion =     torch.nn.CrossEntropyLoss().cuda()    # Loss function
    else:
        print("Undefined Loss Function")
        exit()

    params  = list(model.parameters())                                      # Parameters to train


    # Optimizer -- the optimizer is not specified in the paper, and was assumed to
    # be SGD. The parameters of the model were also not specified and were set
    # to commonly values used by pytorch (lr = 0.1, momentum = 0.3, decay = 1e-4)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    # The paper does not specify an annealing factor, we set it to 1.0 (no annealing)
    scheduler = MultiStepLR( optimizer,
                             milestones=list(range(0, args.num_epochs, 1)),
                             gamma=args.annealing_factor)


    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()


    if args.experiment_type == "exp1":
    
        print("#############  Start Training (Experiment 1)    ##############")
        total_step = len(train_loader)
    
    
        cluster_centers, cluster_assignment = indexing_step(model = model, data_loader = train_loader, cluster_centers = None)
        
        loss_vector = 10. * np.ones(args.K * args.num_classes)
        loss_count  = np.ones(args.K * args.num_classes)

        for epoch in range(0, args.num_epochs):
    
            if args.evaluate_only:
                exit()
            order = define_order(cluster_assignment, cluster_centers, loss_vector/loss_count)         
            train_loader.dataset.update_read_order(order)
    
    
            logger.add_scalar("Misc/Epoch Number", epoch, epoch * total_step)
            
            loss_vector, loss_count, stdev = train_step(   model = model,         
                                        train_loader = train_loader,
                                        criterion    = criterion,
                                        epoch        = epoch,
                                        optimizer    = optimizer,
                                        step         = epoch * total_step,
                                        valid_loader = valid_loader,
                                        assignment   = cluster_assignment,
                                        loss_vector  = loss_vector,
                                        loss_count   = loss_count)
    
            if args.optimizer == 'sgd':    
                scheduler.step()
    
            if isinstance(loss_vector, np.ndarray):
                loss_value = loss_vector.mean().item()
            elif isinstance(loss_vector, list):
                loss_value = np.mean(loss_vector)  # Use np.mean to ensure compatibility
            else:
                raise TypeError("Unsupported type for loss_vector: {}".format(type(loss_vector)))
            
            # Now plot the loss
    
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT loss_value = ", loss_value)
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT stdev.item() = ", stdev.item())
            plotter.plot('loss', 'loss train_step', epoch, loss_value)
            
            plotter.plot('stdev', 'stdev train_step',epoch, stdev.item())
            
            logger.add_scalar(args.dataset + "/STDEV ",   stdev,   epoch * total_step)
    
            if epoch % 3 == 0:
                curr_loss, curr_wacc = eval_step(   model       = model,                     
                                                    data_loader = train_loader,
                                                    criterion   = criterion,
                                                    step        = epoch * total_step,
                                                    datasplit   = "train",
                                                    stdev       = stdev,
                                                    cluster_centers = cluster_centers)
    
    
                curr_loss, curr_wacc = eval_step(   model       = model,                    
                                                    data_loader = valid_loader,
                                                    criterion   = criterion,
                                                    step        = epoch * total_step,
                                                    datasplit   = "valid",
                                                    stdev       = stdev,
                                                    cluster_centers = cluster_centers)
    
            cluster_centers, assignment = indexing_step( model = model, data_loader = train_loader, cluster_centers = cluster_centers) 
            if epoch % 5 == 0:
                args = save_checkpoint(  model      = model,
                                         optimizer  = optimizer,
                                         curr_epoch = epoch,
                                         curr_loss  = curr_loss,
                                         curr_step  = (total_step * epoch),
                                         args       = args,
                                         curr_acc   = curr_wacc,
                                         filename   = ('model@epoch%d.pkl' %(epoch)))
    
        # Final save of the model
        args = save_checkpoint(  model      = model,
                                 optimizer  = optimizer,
                                 curr_epoch = epoch,
                                 curr_loss  = curr_loss,
                                 curr_step  = (total_step * epoch),
                                 args       = args,
                                 curr_acc   = curr_wacc,
                                 filename   = ('model@epoch%d.pkl' %(epoch)))
        

    if args.experiment_type == "exp2":
        print("#############  Start Training (Experiment 2)    ##############")
        
        class_sets = [ [0, 1],  [2, 3], [4, 5],  [6, 7],  [8, 9] ]
        args.num_classes = 2
        # Loop through each increment
        for increment in range(len(class_sets)):
            print(f"Training increment {increment + 1} with classes {class_sets[increment]}")
            args.current_classes = class_sets[increment]
            train_loader.dataset.get_classes(class_sets[increment])
            total_step = len(train_loader)

            cluster_centers, cluster_assignment = indexing_step(model = model,data_loader = train_loader, cluster_centers = None)
            
            loss_vector = 10. * np.ones(args.K * args.num_classes)
            loss_count  = np.ones(args.K * args.num_classes)

            
            for epoch in range(args.num_epochs):
                order = define_order(cluster_assignment, cluster_centers, loss_vector/loss_count)         
                train_loader.dataset.update_read_order(order)    
        
                logger.add_scalar("Misc/Epoch Number", epoch, epoch * total_step)
                
                loss_vector, loss_count, stdev = train_step(   model = model,         
                                            train_loader = train_loader,
                                            criterion    = criterion,
                                            epoch        = epoch,
                                            optimizer    = optimizer,
                                            step         = epoch * total_step,
                                            valid_loader = None,
                                            assignment   = cluster_assignment,
                                            loss_vector  = loss_vector,
                                            loss_count   = loss_count) 
                if args.optimizer == 'sgd':
                    scheduler.step()
                cluster_centers, assignment = indexing_step( model = model, data_loader = train_loader, cluster_centers = cluster_centers)
                
            # After training the current increment, evaluate on all previous increments
            for eval_increment in range(increment + 1):
                # Update the dataset to include all classes up to the current increment
                eval_classes = [cls for sublist in class_sets[:eval_increment + 1] for cls in sublist]
                print("evaluation on eval_classes = ",eval_classes)
                valid_loader.dataset.get_classes(eval_classes)
                total_step = len(valid_loader)
                # print(len(valid_loader))
    
                # Evaluate the model
                eval_loss, eval_accuracy = eval_step(
                    model=model,
                    data_loader=valid_loader,
                    criterion=criterion,
                    step= epoch * total_step,
                    datasplit='valid',
                    stdev=stdev,
                    cluster_centers=cluster_centers
                )
    
                print(f"Evaluation after increment {increment + 1} on classes {eval_classes}: Loss = {eval_loss}, Accuracy = {eval_accuracy}")
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT Finish : increment ", increment)
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT Training completed for all increments TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT.")


#####################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--save_epoch',         type=int , default=10)
    parser.add_argument('--eval_epoch',         type=int , default=3)
    parser.add_argument('--eval_step',          type=int , default=1000)
    parser.add_argument('--log_rate',           type=int, default=10)
    parser.add_argument('--workers',            type=int, default=7)
    parser.add_argument('--world_size',         type=int, default=1)

    # parser.add_argument('--num_epochs',         type=int,   default=50)
    parser.add_argument('--num_epochs',         type=int,   default=2)
    parser.add_argument('--embedding_size',     type=int,   default=512)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight_decay',       type=float, default=1e-4)
    parser.add_argument('--optimizer',          type=str,   default='sgd')
    parser.add_argument('--annealing_factor',   type=float, default=1.0)

    parser.add_argument('--dataset',            type=str, default='cifar10')
    parser.add_argument('--model',              type=str, default='inception')
    parser.add_argument('--experiment_name',    type=str, default= 'MagnetTest')
    parser.add_argument('--loss',               type=str, default= 'magnet')
    # parser.add_argument('--experiment_type',    type=str, default= 'exp1') #exp2 for experiment2
    parser.add_argument('--experiment_type',    type=str, default= 'exp2')
    parser.add_argument('--evaluate_only',      action="store_true",default=False)
    parser.add_argument('--evaluate_train',     action="store_true",default=False)
    parser.add_argument('--resume',             type=str, default=None)

    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

    # Magnet Loss Parameters 
    parser.add_argument('--M',                  type=int, default=8)       # Number of nearest clusters per mini-batch
    parser.add_argument('--K',                  type=int, default=8)       # Number of clusters per class
    parser.add_argument('--L',                  type=int, default=8)       # Number of clusters per class
    # parser.add_argument('--D',                  type=int, default=4)       # Number of examples per cluster
    parser.add_argument('--D',                  type=int, default=2)       # Number of examples per cluster
    parser.add_argument('--GAP',                type=float, default=4)       # Number of examples per cluster
    parser.add_argument('--NUM_BATCHES',        type=int, default=100)       # Number of examples per cluster

    parser.add_argument('--visdom', dest='visdom', action='store_true', default= True, help='Use visdom to track and plot')
    parser.add_argument('--name', default='MagnetLoss', type=str,help='name of experiment')
    parser.add_argument('--current_classes',    type=list, default= [])
  # args.current_classes  
    args = parser.parse_args()
    print(args)
    print("")

    args.batch_size = args.M * args.D

    if args.dataset == "cifar10":
        args.data_path = None
        args.num_classes = 10

    root_dir                    = os.path.dirname(os.path.abspath(__file__))
    experiment_result_dir       = os.path.join(root_dir, os.path.join('experiments',args.dataset))
    args.full_experiment_name   = ("exp_%s_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), args.dataset, args.experiment_name) )
    args.experiment_path        = os.path.join(experiment_result_dir, args.full_experiment_name)
    args.best_loss              = sys.float_info.max
    args.best_acc               = 0.

    # Create model directory
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    print("Experiment path is : ", args.experiment_path)

    if args.M > (args.K * args.num_classes):
        print("Number of imposter clusters is larger than number of possible clusters. Setting it to NUM_CLUSTERS - 2")
        args.M = (args.K * args.num_classes) - 2


    # Define Logger
    tensorboard_logdir = './tensorboard_logs'
    log_name    = args.full_experiment_name
    logger      = Logger(log_dir = os.path.join(tensorboard_logdir, log_name))

    main(args)

