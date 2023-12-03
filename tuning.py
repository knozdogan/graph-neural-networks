# -*- coding: utf-8 -*-
from functools import partial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torchvision.transforms as T
from torch_geometric.loader import DataLoader
from torchmetrics import Precision, Recall, F1Score

from data import GPRDataset
from transform import ToRAG
from network import GraphNetwork

def load_data(config, data_dir="./data"):
    transform = T.Compose([ToRAG(compactness=config["compactness"], n_segments=config["n_segments"], sigma=1)])
    trainset = GPRDataset('/home/.../gnn/data/train', transform=transform)
    testset = GPRDataset('/home/.../gnn/data/test', transform=transform)

    return trainset, testset


def train_gpr(config, checkpoint_dir=None, data_dir=None):
    net = GraphNetwork(embedding=int(config["embedding"]))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(config, data_dir)

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    testloader = DataLoader(
        testset,
        batch_size=100)

    precision = Precision('binary').to(device)
    recall = Recall('binary').to(device)
    f1score = F1Score('binary').to(device)

    for epoch in range(5):
        net.train()
        running_loss = 0.0
        epoch_step = 0
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()  # Clear gradients.
            out = net(data.x.float(), data.edge_index)  # Perform a single forward pass.
            loss = loss_fn(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()

            running_loss += loss.item()
            epoch_step += 1
            if epoch_step % 10 == 0:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, epoch_step + 1,
                                                running_loss / epoch_step))

        net.eval()
        with torch.no_grad():
            _Map = 0
            _Recall = 0
            _F1Score = 0
            k = 0
            for tbatch in testloader:
                tbatch = tbatch.to(device)
                pred = torch.round(net(tbatch.x.float(), tbatch.edge_index).sigmoid())
                _Map += precision(pred, tbatch.y)
                _Recall += recall(pred, tbatch.y)
                _F1Score += f1score(pred, tbatch.y) 
                k += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(running_loss / epoch_step), f1_score=float(_F1Score/k), recall=float(_Recall/k), ap=float(_Map/k))
    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    
    config = {
        "embedding": tune.choice([32, 64, 128]),
        "compactness": tune.choice([0.1, 1, 10, 100]),
        "n_segments": tune.choice([256, 512, 1024, 2048]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    
    load_data(config, data_dir)
    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["f1_score", "recall", "ap"])
    result = tune.run(
        partial(train_gpr, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("f1_score", "max", "last")
    print("Best trial config: {}".format(best_trial))

    best_trained_model = GraphNetwork()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
