from param import parameter_parser
from MMGCN import MMGCN
from dataprocessing import data_pro
import torch


def train(model, train_data, optimizer, opt):
    model.train()
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score = model(train_data)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'].cuda())
        loss.backward()
        optimizer.step()
        print(loss.item())
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin)
    return score

def main():
    args = parameter_parser()
    dataset = data_pro(args)
    train_data = dataset
    model = MMGCN(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    score = train(model, train_data, optimizer, args)

if __name__ == "__main__":
    main()
