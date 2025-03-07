import os

temp_root = {
    dir_type: [
            os.path.join('temp', dir_type, f'party-{idx}') for idx in range(10)
        ] for dir_type in ['file', 'model']
    }

passive_list = []
pp_list=[]
pp=None
pub_key_list=[]
key_list=[[]]

stub_list=[None,None]

if __name__ == '__main__':
    print(temp_root)
    