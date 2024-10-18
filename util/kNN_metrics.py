"""
    A set of evaluation metrics for clustering

    NMI, Purity and RI are calculated using equations in the following link:
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Hungarian Algorithm obtained from https://github.com/tdedecko/hungarian-algorithm

"""

import numpy    as np
import time

from sklearn    import cluster
from IPython    import embed

# class softkNN_metrics(object):

#     def __init__(self, sigma = 1.0, num_clusters = 37):
#         self.reset()
#         self.sigma          = sigma
#         self.num_clusters    = num_clusters

#     def reset(self):
#         self.embeddings = []
#         self.labels     = []

#     def update(self, embeddings, labels):
#         self.embeddings += list(embeddings.cpu().data.numpy())
#         self.labels     += list(labels.numpy())


#     def get_softmax_probs(self, stdev):
#         # First calculate an N^2 matrix for distances
#         softmax_probs = np.zeros((len(self.embeddings), self.num_clusters))
#         distances = np.zeros((len(self.embeddings), len(self.embeddings)))

#         # Inefficient for now,
#         # assuming potentially having 2 sets of embeddings (one annotated, and one is not)
#         for i in range(0, len(self.embeddings)-2):
#             for j in range(i+1, len(self.embeddings)):
#                 d = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
#                 distances[i][j] = d
#                 distances[j][i] = d


#         for i in range(0, len(self.embeddings)):
#             elements = list(np.argsort(distances[i])[0:129])
#             elements.remove(i)

#             # find the top 128 elements
#             close_distances = distances[i][elements]
#             close_labels    = [self.labels[e] for e in elements]

#             # calculate softmax values for all and sum for each
#             for j in range(0, len(close_labels)):
#                 softmax_probs[i][close_labels[j]] += np.exp(-0.5 * close_distances[j] / stdev)

#             softmax_probs[i] = softmax_probs[i] / np.sum(softmax_probs[i])

#         return softmax_probs

#     def accuracy(self, stdev = 1.0):
#         cur_time = time.time()
#         softmax_probs = self.get_softmax_probs(stdev = stdev)
#         print("Evaluation: Softmax Probability Calculation (Time Elapsed {time:.3f})".format(time=time.time() - cur_time))
#         acc1, acc5 = 0., 0.
#         # Calculate top1/top5/ etc etc
#         for i in range(0, softmax_probs.shape[0]):
#             order = list(np.argsort(softmax_probs[i])[::-1])
#             acc1 += float(self.labels[i] == order[0])
#             acc5 += float(self.labels[i] in order[0:5])


#         return 100.0*acc1/float(softmax_probs.shape[0]), 100.0*acc5/float(softmax_probs.shape[0])



class softkNN_metrics(object):

    def __init__(self, sigma=1.0, num_clusters=37):
        self.reset()
        self.sigma = sigma
        self.num_clusters = num_clusters

    def reset(self):
        self.embeddings = []
        self.labels = []

    def update(self, embeddings, labels):
        print(f"Updating metrics with {embeddings.size()} embeddings and {labels.size()} labels")
        self.embeddings += list(embeddings.cpu().data.numpy())
        self.labels += list(labels.numpy())

    def get_softmax_probs(self, stdev):
        if len(self.embeddings) == 0:
            print("No embeddings found. Returning empty softmax probabilities.")
            return np.array([])

        softmax_probs = np.zeros((len(self.embeddings), self.num_clusters))
        distances = np.zeros((len(self.embeddings), len(self.embeddings)))

        for i in range(len(self.embeddings) - 1):
            for j in range(i + 1, len(self.embeddings)):
                d = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                distances[i][j] = d
                distances[j][i] = d

        for i in range(len(self.embeddings)):
            elements = list(np.argsort(distances[i])[:129])
            elements.remove(i)

            close_distances = distances[i][elements]
            close_labels = [self.labels[e] for e in elements]

            for j in range(len(close_labels)):
                softmax_probs[i][close_labels[j]] += np.exp(-0.5 * close_distances[j] / stdev)

            softmax_probs[i] = softmax_probs[i] / np.sum(softmax_probs[i]) if np.sum(softmax_probs[i]) > 0 else 1e-10

        return softmax_probs

    def accuracy(self, stdev=1.0):
        cur_time = time.time()
        softmax_probs = self.get_softmax_probs(stdev=stdev)
        print("Evaluation: Softmax Probability Calculation (Time Elapsed {time:.3f})".format(time=time.time() - cur_time))

        if softmax_probs.shape[0] == 0:
            print("No embeddings found. Skipping accuracy calculation.")
            return 0.0, 0.0

        acc1, acc5 = 0.0, 0.0
        for i in range(softmax_probs.shape[0]):
            order = list(np.argsort(softmax_probs[i])[::-1])
            acc1 += float(self.labels[i] == order[0])
            acc5 += float(self.labels[i] in order[0:5])

        return 100.0 * acc1 / float(softmax_probs.shape[0]), 100.0 * acc5 / float(softmax_probs.shape[0])





class softkNC_metrics(object):

    def __init__(self, stdev, cluster_centers, L=128, K = 4):
        self.reset()
        self.stdev              = stdev
        self.cluster_centers    = cluster_centers
        self.L                  = min(L, cluster_centers.shape[0])
        self.K                  = K

    def reset(self):
        self.embeddings = []
        self.labels     = []

    def update(self, embeddings, labels):
        self.embeddings += list(embeddings.cpu().data.numpy())
        self.labels     += list(labels.numpy())


    def get_softmax_probs(self, stdev):
        # First calculate an N^2 matrix for distances
        softmax_probs = np.zeros((len(self.embeddings), self.cluster_centers.shape[0] // self.K))
        distances = np.zeros((len(self.embeddings), self.cluster_centers.shape[0] ))

        # Inefficient for now,
        # assuming potentially having 2 sets of embeddings (one annotated, and one is not)
        for i in range(0, len(self.embeddings)):
            for j in range(0, self.cluster_centers.shape[0]):
                d = np.linalg.norm(self.embeddings[i] - self.cluster_centers[j])
                distances[i][j] = d

        for i in range(0, len(self.embeddings)):
            elements = list(np.argsort(distances[i])[0:self.L])

            # find the top 128 elements
            close_distances = distances[i][elements]
            close_labels    = [e//self.K for e in elements]

            # calculate softmax values for all and sum for each
            for j in range(0, len(close_labels)):
                softmax_probs[i][close_labels[j]] += np.exp(-0.5 * close_distances[j] / stdev)

            softmax_probs[i] = softmax_probs[i] / np.sum(softmax_probs[i])

        return softmax_probs

    def accuracy(self, stdev = 1.0):  #original
        cur_time = time.time()
        softmax_probs = self.get_softmax_probs(stdev = self.stdev)
        print("Evaluation: Softmax Probability Calculation (Time Elapsed {time:.3f})".format(time=time.time() - cur_time))
        acc1, acc5 = 0., 0.
        # Calculate top1/top5/ etc etc
        for i in range(0, softmax_probs.shape[0]):
            order = list(np.argsort(softmax_probs[i])[::-1])
            acc1 += float(self.labels[i] == order[0])
            acc5 += float(self.labels[i] in order[0:5])

        return 100.0*acc1/float(softmax_probs.shape[0]), 100.0*acc5/float(softmax_probs.shape[0]) 

    # def accuracy(self, stdev=1.0):
    #     cur_time = time.time()
        
    #     # Check if embeddings are available before calculating softmax probabilities
    #     if len(self.embeddings) == 0:
    #         print("No embeddings available for accuracy calculation.")
    #         return 0.0, 0.0
        
    #     softmax_probs = self.get_softmax_probs(stdev=self.stdev)
        
    #     print("Evaluation: Softmax Probability Calculation (Time Elapsed {time:.3f})".format(time=time.time() - cur_time))
        
    #     acc1, acc5 = 0., 0.
        
    #     # Calculate top1/top5/ etc etc
    #     for i in range(softmax_probs.shape[0]):
    #         order = list(np.argsort(softmax_probs[i])[::-1])
    #         acc1 += float(self.labels[i] == order[0])
    #         acc5 += float(self.labels[i] in order[0:5])
    
    #     total_samples = float(softmax_probs.shape[0])
    #     return (100.0 * acc1 / total_samples if total_samples > 0 else 0), (100.0 * acc5 / total_samples if total_samples > 0 else 0)
