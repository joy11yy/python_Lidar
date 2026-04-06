
import h5py

l2a_file = r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024243075949_O32374_02_T06551_02_004_04_V002_subsetted.h5"

with h5py.File(l2a_file, 'r') as f:
    print("=== 顶层对象 ===")
    for key in f.keys():
        print(f"  {key}")

        # 如果是波束分组，再打印里面的内容
        if key.startswith('BEAM'):
            print(f"    {key} 下的对象:")
            for subkey in f[key].keys():
                print(f"      - {subkey}")