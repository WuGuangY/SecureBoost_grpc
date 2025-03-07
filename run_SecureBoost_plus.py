import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.ActiveParty_SecureBoost_plus import ActiveParty
from core.PassivePartty_SecureBoost_plus import PassiveParty

from utils.params import pp_list,pp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--number', dest='passive_num', type=int, default=1)
    args = parser.parse_args()

    passive_num = args.passive_num

    ppid=1
    
    #创建被动方
    path_list = [
        f'temp/file/party-{ppid}', 
        f'temp/model/party-{ppid}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        
    pp = PassiveParty(ppid)
    pp.load_dataset(f'./static/data/train_pp.csv')
    pp_list.append(pp)

    #创建主动方
    id = 0
    path_list = [
        f'temp/file/party-{id}', 
        f'temp/model/party-{id}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    ap = ActiveParty() 
    ap.load_dataset('./static/data/train.csv')
    ap.train()
    file_name = ap.dump_model('./static/model/')
    ap.load_model(file_name)
    ap.predict()