import torch
from dataloader import DataLoader
from model import Model
from dice_loss import DiceLoss
import warnings

warnings.filterwarnings("ignore")


def valid(valid_loader, model):
    model.eval()
    criterion = DiceLoss()
    for iter, (x, y) in enumerate(valid_loader):
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y)
            acc = ((outputs > 0) == y).sum(dim=0).float() / VALID_BATCH_SIZE
    
    mean_acc = acc.mean()
    output_log = '(Valid)  Loss: {loss:.3f} | Mean Acc: {acc:.3f}'.format(
        loss=loss.item(),
        acc=mean_acc.item()
    )
    print(output_log)
    print(acc)
    return mean_acc

def main():
    valid_loader = torch.utils.data.DataLoader(
        DataLoader(split="valid"), batch_size=VALID_BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False, pin_memory=True
    )
    model = Model().cuda()
    state_dict = torch.load("checkpoint/checkpoint.pth")
    model.load_state_dict(state_dict)
    valid(valid_loader, model)


if __name__ == '__main__':
    VALID_BATCH_SIZE = 198
    main()