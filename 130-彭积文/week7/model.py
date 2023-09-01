import torch.nn as nn
from transformers import BertModel


class BertModule(nn.Module):

    def __init__(self,config):
        super(BertModule,self).__init__()
        self.__config = config
        self.__bert = BertModel.from_pretrained("../bert", return_dict=False).to(self.__config["device"])
        self.__linear1 = nn.Linear(self.__config["input_dim"], self.__config["hidden_dim"]).to(self.__config["device"])
        self.__activation = nn.Sigmoid()
        self.__linear2 = nn.Linear(self.__config["hidden_dim"], self.__config["output_dim"]).to(self.__config["device"])
        self.__dropout = nn.Dropout(0.2)
        self.__loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,x,y=None):
        _,x = self.__bert(x)
        x = self.__linear1(x)
        x = self.__activation(x)
        x = self.__linear2(x)
        y_pred = self.__dropout(x)

        if y is not None:
            return self.__loss(y_pred, y.squeeze())
        else:
            return y_pred