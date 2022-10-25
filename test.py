import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from RS_Zoos.util.batch_test import data_generator, test
from RS_Zoos.util.helper import set_seed
from RS_Zoos.model.LightGCN import LightGCN
from RS_Zoos.model.NGCF import NGCF
from RS_Zoos.util.parser import args

weight_file = f"./checks/clean_{args.dataset}_{args.model}_{args.pretrain}.pkl"
plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()  # 'norm_adj' for LightGCN; 'mean_adj' for NGCF
set_seed(2022)
args.device = torch.device('cuda:' + str(args.gpu_id))

# read the model
# model = NGCF(data_generator.n_users,
#              data_generator.n_items,
#              norm_adj,
#              args).to(args.device)

if args.pretrain == 1:
    path = args.data_path + args.dataset + 's_'
    args.user_emb = np.load(path + args.user_emb + '.npy', allow_pickle=True)
    args.item_emb = np.load(path + args.item_emb + '.npy')
else:
    args.user_emb = None
    args.item_emb = None

model = LightGCN(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

if __name__ == '__main__':
    model.load_state_dict(torch.load(weight_file))

    # test the model
    users_to_test = list(data_generator.test_set.keys())
    ret0 = test(model, users_to_test, drop_flag=False)
    print(ret0)

    # # read plot/yelp2018_LightGCN_0.csv
    # df1 = pd.read_csv('plot/yelp2018_NGCF_0.csv')
    # df2 = pd.read_csv('plot/yelp2018_LightGCN_0.csv')
    # df3 = pd.read_csv('plot/yelp2018_LightGCN_1.csv')
    #
    # # plot
    # plt.figure(figsize=(10, 6))
    # sns.set_theme(style="ticks")
    # sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    # sns.despine()
    #
    # plt.plot(df1['epoch'], df1['loss'], label='NGCF')
    # plt.plot(df2['epoch'], df2['loss'], label='LightGCN')
    # plt.plot(df3['epoch'], df3['loss'], label='LightGCN+')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('loss on Yelp2018_10k')
    # plt.legend()
    # plt.show()
