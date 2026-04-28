import h5py
from pathlib import Path
def explore_gedi_file(filepath):
    """探索GEDI H5文件结构"""
    print("=" * 70)
    print(f"探索GEDI文件: {Path(filepath).name}")
    print("=" * 70)

    with h5py.File(filepath, 'r') as f:
        print(f"\n顶层keys: {list(f.keys())}")

        # 查找波束组
        beams = [k for k in f.keys() if 'BEAM' in k.upper()]
        print(f"\n波束组: {beams}")

        for beam in beams[:2]:  # 只查看前两个波束作为示例
            print(f"\n{'=' * 50}")
            print(f"波束: {beam}")
            print(f"{'=' * 50}")
            beam_group = f[beam]
            print(f"子keys: {list(beam_group.keys())}")

            # 递归探索每个元素
            for key in beam_group.keys():
                item = beam_group[key]
                print(f"\n  [{key}]")

                # 判断是Dataset还是Group
                if isinstance(item, h5py.Dataset):
                    print(f"    类型: Dataset")
                    print(f"    形状: {item.shape}")
                    print(f"    数据类型: {item.dtype}")
                    if item.size > 0 and item.size < 100:
                        print(f"    示例值: {item[:5]}")
                    elif item.size > 0:
                        print(f"    前5个值: {item[:5]}")
                elif isinstance(item, h5py.Group):
                    print(f"    类型: Group")
                    print(f"    子keys: {list(item.keys())}")
                    # 进一步探索子组的内容
                    for subkey in list(item.keys())[:3]:  # 只显示前3个
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(f"      [{subkey}] Dataset, 形状: {subitem.shape}")
                        else:
                            print(f"      [{subkey}] Group")
                else:
                    print(f"    类型: {type(item)}")

        # 统计总体信息
        print(f"\n{'=' * 70}")
        print("统计信息")
        print(f"{'=' * 70}")

        for beam in beams:
            beam_group = f[beam]
            if 'latitude' in beam_group:
                lats = beam_group['latitude'][:]
                lons = beam_group['longitude'][:]
                print(f"\n{beam}:")
                print(f"  足迹数量: {len(lats)}")
                print(f"  纬度范围: {lats.min():.4f} ~ {lats.max():.4f}")
                print(f"  经度范围: {lons.min():.4f} ~ {lons.max():.4f}")

                # 查看有哪些波形特征
                rh_keys = [k for k in beam_group.keys() if 'rh' in k.lower()]
                if rh_keys:
                    print(f"  波形特征: {rh_keys}")
            else:
                # 如果latitude不在顶层，可能在子组里
                print(f"\n{beam}:")
                for subkey in ['fpdata', 'wavedata']:
                    if subkey in beam_group:
                        subgroup = beam_group[subkey]
                        if 'latitude' in subgroup:
                            lats = subgroup['latitude'][:]
                            lons = subgroup['longitude'][:]
                            print(f"  足迹数量 ({subkey}): {len(lats)}")
                            print(f"  纬度范围: {lats.min():.4f} ~ {lats.max():.4f}")
                            print(f"  经度范围: {lons.min():.4f} ~ {lons.max():.4f}")
                            break

        return beams


# 探索GEDI文件
gedi_file = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5"
beams = explore_gedi_file(gedi_file)