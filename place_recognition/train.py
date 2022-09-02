import torch
from torch.utils.data import Dataset, DataLoader

from network.network import *
from data.data import *

def train():
    num_epochs = 100
    batch_size = 20
    load_model = None
    lr = 0.001
    momentum = 0.9
    save_model = 10
    save_model_path = './placerecognet_model.pth'
    val_ckpt = 10

    # Create Model
    model = PlaceRecogNet()
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    model = model.cuda()

    model.train()

    training_data = RadarLidarDataset(radar_imdir='', odom_file='', map_file='',
                                      rtime_file='', radar_transform='', map_transform='')
    validation_data = RadarLidarDataset(radar_imdir='', odom_file='', map_file='',
                                        rtime_file='', radar_transform='', map_transform='')

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch_idx, epoch in enumerate(range(num_epochs)):
        print(f"Training epoch : {epoch_idx}")
        for anchor, pos_sample, neg_sample in train_loader:
            anchor = anchor.cuda()
            pos_sample = pos_sample.cuda()
            neg_sample = neg_sample.cuda()

            out_anchor, out_pos, out_neg = model(anchor, pos_sample, neg_sample)

            loss = criterion(out_anchor, out_pos, out_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch_idx % save_model == 0:
                torch.save(model.state_dict(), save_model_path)

            # Validation
            # if epoch_idx % val_ckpt == 0:
            #     for val_anchor, val_pos_sample, val_neg_sample in val_loader:
            #         val_anchor = val_anchor.cuda()
            #         val_pos_sample = val_pos_sample.cuda()
            #         val_neg_sample = val_neg_sample.cuda()
            #
            #         val_out_anchor, val_out_pos, val_out_neg = model(val_anchor, val_pos_sample, val_neg_sample)

if __name__ == '__main__':
    train()


