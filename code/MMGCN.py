import torch
from torch import nn
from torch_geometric.nn import GCNConv
torch.backends.cudnn.enabled = False

class MMGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MMGCN, self).__init__()
        self.args = args
        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)

        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))
        self.fc1_x = nn.Linear(in_features=self.args.view*self.args.gcn_layers,
                             out_features=5*self.args.view*self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5*self.args.view*self.args.gcn_layers,
                             out_features=self.args.view*self.args.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                             out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                             out_features=self.args.view * self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.view*self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        x_m = torch.randn(self.args.miRNA_number, self.args.fm)
        x_d = torch.randn(self.args.disease_number, self.args.fd)


        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        y_d_f1 = torch.relu(self.gcn_y1_f(x_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f2 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))

        y_d_s1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))

        XM = torch.cat((x_m_f1, x_m_f2, x_m_s1, x_m_s2), 1).t()

        XM = XM.view(1, self.args.view*self.args.gcn_layers, self.args.fm, -1)

        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM

        XM_channel_attention = torch.relu(XM_channel_attention)

        YD = torch.cat((y_d_f1, y_d_f2, y_d_s1, y_d_s2), 1).t()

        YD = YD.view(1, self.args.view*self.args.gcn_layers, self.args.fd, -1)

        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,1)
        YD_channel_attention = y_channel_attenttion * YD

        YD_channel_attention = torch.relu(YD_channel_attention)



        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()



        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()


        return x.mm(y.t())










