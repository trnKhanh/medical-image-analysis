import os
import random
from utils.dataloader import *
from utils.augmentation import *
import random
from utils.criterion import MyCriterion
from torch.utils.data import DataLoader
from utils.metrics import DSC, HD
from models.unet import U_Net


def get_model():
    model = U_Net()
    return model


def get_dataloader(batch_size=2):
    tf_train = JointTransform2D(crop=None, p_flip=0.5,p_rota=0.5,long_mask=True)
    tf_val = JointTransform2D(crop=None, long_mask=True)
    ls = os.listdir("./dataset/train/images")
    random.shuffle(ls)
    length = len(ls)
    ratio = 0.8
    number_train = int(length * ratio)

    train_set = DatasetSegmentation("./dataset/train", ls[:number_train],tf_train)
    val_set = DatasetSegmentation("./dataset/train", ls[number_train:],tf_val)
    dataset_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return dataset_train, dataset_val


def save_training_process(train_losses, val_losses, start_val_epoch):
    with open("./process.txt", "w") as file:
        num_epochs = len(train_losses)
        for i in range(num_epochs):
            file.write(f"Epoch {i} Train Loss: {train_losses[i]}")
            if (i + 1) >= start_val_epoch:
                index = i - (start_val_epoch - 1)
                file.write(f"Epoch {i} Val DSC: {val_losses[index]}")
            file.write("\n")


def main():
    # ========================== set random seed ===========================#
    seed_value = 2024  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    # ========================== set hyper parameters =========================#
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    BASE_LEARNING_RATE = 0.0001
    START_VAL_EPOCH = 0

    # ========================== get model, dataloader, optimizer and so on =========================#
    model = get_model()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LEARNING_RATE,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset_train, dataset_val = get_dataloader(BATCH_SIZE)

    criterion = MyCriterion()  # combined loss function
    evalue = HD()  # metric to find best model

    best_val_hd = 700
    train_losses = []
    val_hd = []

    # ========================== training =============================#
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(dataset_train):
            print(f"Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch {batch_idx}: ", end="")
            batch_train_loss = []
            imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS, 3, 336, 544) float32
            masks = sample['label'].to(device=device)  # (BS,336,544)  int64  0: background 1:ps 2:fh

            preds = model(imgs)  # (BS, 3, 336, 544)
            train_loss = criterion(preds, masks)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_train_loss.append(train_loss.detach().cpu().numpy())

        train_losses.append(np.mean(batch_train_loss))

        # ========================= validation =======================#
        if (epoch + 1) >= START_VAL_EPOCH:
            model.eval()
            hd_ls = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(dataset_val):
                    imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS,3,512,512)
                    masks = sample['label'].to(device=device)  # (BS,512,512)
                    preds = model(imgs)
                    # dc1,dc2 = evalue(preds, masks)
                    hd  = evalue(preds, masks)
                    #dc_all = (dc1 + dc2)/2
                    
                    hd_ls.append(hd)
                hd_mean = np.mean(hd_ls)
                print(f"Validation|| Epoch[{epoch + 1}/{NUM_EPOCHS}] Batch {batch_idx}: hd: {hd_mean:.6f}")
                val_hd.append(hd_mean)

                # ================  SAVING ================#
                if hd_mean < best_val_hd:
                    best_val_hd = hd_mean
                    torch.save(model.state_dict(), f"./checkpoints/epoch{epoch}_val_{hd_mean:.6f}.pth")

    save_training_process(train_losses, val_hd, START_VAL_EPOCH)


if __name__ == '__main__':
    main()