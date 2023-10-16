import os
import torch
import time
# a nice training progress bar
from tqdm import tqdm
import PointNets_model as pt
import data_processing as ds
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

labels = ds.labels

def compute_stats(true_labels, pred_labels):
  unk     = np.count_nonzero(true_labels == 0)
  trav    = np.count_nonzero(true_labels == 1)
  nontrav = np.count_nonzero(true_labels == 2)

  total_predictions = labels.shape[1]*labels.shape[0]
  correct = (true_labels == pred_labels).sum().item()

  return correct, total_predictions



if __name__ == '__main__':
    # create a new instantiation of PointNetSeg model
    pointnet = pt.PointNetSeg()

    # load pyTorch model weights
    model_path = os.path.join('', "pointnetmodel.yml")
    pointnet.load_state_dict(torch.load(model_path))

    # move the model to cuda
    pointnet.to(device)
    pointnet.eval()
    test_ds   = ds.PointCloudData('dataset', start=120, end=150)
    test_loader   = ds.DataLoader( dataset=test_ds,   batch_size=1, shuffle=False )
    total_correct_predictions = total_predictions = 0

    start = time.time()

    for i, data in tqdm(enumerate(test_loader, 0)):
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, predicted = torch.max(outputs.data, 1)

        # visualize results
        remapped_pred = ds.remap_to_bgr(predicted[0].cpu().numpy(), ds.remap_color_scheme)
        np_pointcloud = inputs[0].cpu().numpy()
        # visualize3DPointCloud(np_pointcloud, remapped_pred)

        # compute statistics
        ground_truth_labels = labels.cpu()
        predicted_labels    = predicted.cpu()
        correct, total = compute_stats(ground_truth_labels, predicted_labels)

        total_correct_predictions += correct
        total_predictions         += total

    end = time.time()

    # nice layout after tqdm
    print()
    print()

    test_acc    = 100. * total_correct_predictions / total_predictions
    tot_latency = end-start
    avg_latency = tot_latency / len(test_loader.dataset)

    print('Test accuracy:', test_acc, "%")
    print('total time:',    tot_latency, " [s]")
    print('avg time  :',    avg_latency, " [s]")
