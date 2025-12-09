import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights, get_padding


class Encoder_cover(nn.Module):
    def __init__(self):
        super(Encoder_cover, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose1d(80, 512, 3, 1, 1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(512, 512, 3, 1, 1, get_padding(3, 1)),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(512, 512, 3, 1, 5, get_padding(3, 5)),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(512))
        self.lstm1 = nn.LSTM(512, 40, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x)
        outputs = outputs.transpose(1, 2)
        return outputs


class Encoder_ste(nn.Module):
    def __init__(self):
        super(Encoder_ste, self).__init__()
        self.conv = nn.Sequential(nn.ConvTranspose1d(240, 1024, 3, 1, 1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(1024))

        self.conv1 = nn.Sequential(nn.ConvTranspose1d(240, 1024, 4, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.ConvTranspose1d(1024, 1024, 4, 1, 2),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm1d(1024))

        self.conv2 = nn.Sequential(nn.ConvTranspose1d(240, 1024, 5, 1, 2),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm1d(1024))

        self.lstm = nn.LSTM(1024, 40, 1, batch_first=True, bidirectional=True)
        self.conv3 = nn.Conv1d(240, 80, 3, 1, 1)

        # self.conv4 = nn.Conv1d(1024*3, 80, 3, 1, 1)
    def forward(self, x):
        '''
        x1 = self.conv(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)
        outputs = torch.cat((x1, x2, x3), dim=1)
        # print(outputs.size())
        outputs = self.conv4(outputs)'''


        x1 = self.conv(x)
        x1 = x1.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs1, _ = self.lstm(x1)
        x2 = self.conv1(x)
        x2 = x2.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs2, _ = self.lstm(x2)
        x3 = self.conv2(x)
        x3 = x3.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs3, _ = self.lstm(x3)
        outputs = torch.cat((outputs1, outputs2, outputs3), dim=2)
        outputs = outputs.transpose(1, 2)
        outputs = self.conv3(outputs)

        return outputs


class Deconder_msg(nn.Module):
    def __init__(self):
        super(Deconder_msg, self).__init__()

        self.conv = nn.Sequential(nn.ConvTranspose1d(80, 1024, 3, 1, 1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                  nn.LeakyReLU(0.1),
                                  nn.BatchNorm1d(1024))

        self.conv1 = nn.Sequential(nn.ConvTranspose1d(80, 1024, 4, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.ConvTranspose1d(1024, 1024, 4, 1, 2),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm1d(1024))

        self.conv2 = nn.Sequential(nn.ConvTranspose1d(80, 1024, 5, 1, 2),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 1, get_padding(3, 1)),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(1024, 1024, 3, 1, 5, get_padding(3, 5)),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm1d(1024))

        self.conv3 = nn.Sequential(nn.Conv1d(1024 * 3, 80, 3, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(80, 80, 3, 1, 1),
                                   nn.LeakyReLU(0.1))

        self.conv4 = nn.Sequential(nn.Conv1d(1024 * 3, 80, 4, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(80, 80, 4, 1, 2),
                                   nn.LeakyReLU(0.1))

        self.conv5 = nn.Sequential(nn.Conv1d(1024 * 3, 80, 5, 1, 2),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(80, 80, 5, 1, 2),
                                   nn.LeakyReLU(0.1))

        self.conv6 = nn.Sequential(nn.Conv1d(240, 80, 3, 1, 1),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv1d(80, 80, 3, 1, 1),
                                   nn.LeakyReLU(0.1))

        self.lstm = nn.LSTM(80, 40, 1, batch_first=True, bidirectional=True)

        self.lstm1 = nn.LSTM(240, 40, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        #print(x.size())
        x1 = self.conv(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)
        x4 = torch.cat((x1, x2, x3), dim=1)

        x5 = self.conv3(x4)
        x6 = self.conv4(x4)
        x7 = self.conv5(x4)

        x8 = torch.cat((x5, x6, x7), dim=1)
        x9 = self.conv6(x8)
        x8 = x8.transpose(1, 2)
        x10, _ = self.lstm1(x8)
        x10 = x10.transpose(1, 2)
        x10 = x10 + x9
        return x10


class stego_model(torch.nn.Module):
    def __init__(self):
        super(stego_model, self).__init__()
        self.rel_serect = Deconder_msg()

    def forward(self, steg):
        rec_serect = self.rel_serect(steg)
        return rec_serect


# if __name__ == '__main__':
    # from config import Config
    #
    # config = Config()
    # device = config.device
    # secret = torch.rand(32, 80, 32).cuda()
    # cover = torch.rand(32, 80, 32).cuda()
    # model = stego_model().to(config.device)
    # C = model(cover, secret, cover, 1)
    # print(C.size())
