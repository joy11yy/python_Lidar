import h5py
import numpy as np


def print_gedi_structure(file_path):
    """
    简单打印GEDI HDF5文件的数据结构
    """

    def print_group(name, obj):
        indent = name.count('/') * '  '
        #isinstance检查一个对象是不是属于某个数据类型
        if isinstance(obj, h5py.Dataset):
            # 数据集：显示名称、形状和数据类型
            print(f"{indent}📊 {name.split('/')[-1]} : shape={obj.shape}, dtype={obj.dtype}")
        else:
            # 组：只显示名称
            print(f"{indent}📁 {name.split('/')[-1]}/")

    with h5py.File(file_path, 'r') as f:
        print(f"\n文件: {file_path}")
        print("=" * 50)
        f['BEAM0001'].visititems(print_group)


# 使用
gedi_L1B_file = r"D:\研究生\PoYangData\GEDI01_B_2025045050211_O34978_03_T09361_02_006_02_V002_subsetted.h5"
print_gedi_structure(gedi_L1B_file)
gedi_L2A_file=r"D:\研究生\PoYangData\GEDI02_A_2025045050211_O34978_03_T09361_02_004_02_V002_subsetted.h5"
print("···················GEDI_L2A···············")
print_gedi_structure(gedi_L2A_file)