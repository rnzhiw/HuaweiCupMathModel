import torch
from dataloader import DataLoader
from model import Model
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def test(valid_loader, model):
    model.eval()

    smiles = pd.read_csv('data/Molecular_Descriptor.csv')['SMILES'].tolist()
    y_preds = []
    for iter, x in enumerate(valid_loader):
        x = x.cuda()
        with torch.no_grad():
            outputs = model(x)
            y_pred = (outputs > 0).int().cpu().numpy().tolist()
            y_preds.append(y_pred)
    y_preds = y_preds[0]
    print(len(y_preds))
    f = open("data/ADEMT_test_pre.csv", "w+")
    f.write("SMILES,Caco-2,CYP3A4,hERG,HOB,MN\n")
    for index, y_pred in enumerate(y_preds):
        text = smiles[index] + "," + ",".join([str(i) for i in y_pred])
        f.write(text + "\n")
        print(text)
    f.close()


def main():
    valid_loader = torch.utils.data.DataLoader(
        DataLoader(split="test"), batch_size=50,
        shuffle=False, num_workers=0, drop_last=False, pin_memory=True
    )
    model = Model().cuda()
    state_dict = torch.load("checkpoint/checkpoint.pth")
    model.load_state_dict(state_dict)
    test(valid_loader, model)


if __name__ == '__main__':
    main()