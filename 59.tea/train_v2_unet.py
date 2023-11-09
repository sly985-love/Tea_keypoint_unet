import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasets_v2 import KeyPointDatasets_6P
from models import unet
# from utils import Visualizer, compute_loss
from utils.utils import compute_loss
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# h, w
IMG_SIZE = 256, 256

# vis = Visualizer(env="keypoint")
# trian就是最基本最常态的东西，与其他东西没有任何不同

def train(model, epoch, dataloader, optimizer, criterion, scheduler):
    model.train()
    totalLoss = 0
    for itr, (image, hm) in enumerate(dataloader):

        if torch.cuda.is_available():
            hm = hm.cuda()
            image = image*255
            image = image.cuda()

        bs = image.shape[0]

        output = model(image)

        hm = hm.float()

        # print("output: ", output.shape)
        # print("hm: ", hm.shape)

        loss = criterion(output, hm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if itr % 10 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" %
                  (epoch, itr, loss.item()/bs))
            # vis.plot_many_stack({"train_loss": loss.item()/bs})
        totalLoss += loss.item()/bs
    print("Epoch: {:04d}\tTotalLoss: {}".format(epoch, totalLoss))
    print("\n")


def a_test(model, epoch, dataloader, criterion):
    model.eval()
    sum_loss = 0.
    n_sample = 0
    for itr, (image, hm) in enumerate(dataloader):
        if torch.cuda.is_available():
            hm = hm.cuda()
            image = image.cuda()

        output = model(image)
        hm = hm.float()

        print("output: ", output.shape, hm.shape)

        loss = criterion(output, hm)

        sum_loss += loss.item()
        n_sample += image.shape[0]

    print("TEST: epoch:%02d-->loss:%.6f" % (epoch, sum_loss/n_sample))
    # if epoch > 1:
    #     vis.plot_many_stack({"test_loss": sum_loss/n_sample})
    return sum_loss / n_sample


if __name__ == "__main__":

    total_epoch = 500
    bs = 8
    ########################################
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    #     transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
    #                          std=[0.2479, 0.2475, 0.2485])
    # ])

    datasets = KeyPointDatasets_6P(root_dir="./data/crops_v2", transforms=transforms_all)
    data_loader = DataLoader(datasets, shuffle=True, batch_size=bs, collate_fn=datasets.collect_fn)

    # model = KFSG.KFSGNet()
    model = unet.U_net()
    # model.load_state_dict(torch.load("weights/KFSG/KFSG_75.pth"))
    os.makedirs("weights/U_net_tea", exist_ok=True)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()  # compute_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    for epoch in range(total_epoch):
        train(model, epoch, data_loader, optimizer, criterion, scheduler)
        #loss = a_test(model, epoch, data_loader, criterion)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "weights/U_net_tea/U_net_{}.pth".format(epoch))
