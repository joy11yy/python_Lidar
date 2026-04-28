"""
先复制GEDI文件，然后在副本中添加ICESat-2光子（修正版）
生成新的含有单光子和GEDI数据的文件
"""

import h5py
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def copy_gedi_file(original_file, copy_suffix='_with_photons'):
    """
    复制GEDI文件
    """
    original_path = Path(original_file)
    copy_path = original_path.parent / f"{original_path.stem}{copy_suffix}{original_path.suffix}"

    print(f"\n复制文件:")
    print(f"  源文件: {original_file}")
    print(f"  目标文件: {copy_path}")

    if copy_path.exists():
        overwrite = input(f"  文件已存在，是否覆盖？(y/n): ")
        if overwrite.lower() != 'y':
            print("  取消操作")
            return None

    shutil.copy2(original_file, copy_path)
    print(f"  ✓ 复制完成")

    return str(copy_path)


def add_photons_to_gedi_copy(original_gedi_file, icesat2_file,
                             radius_deg=0.00025, output_suffix='_with_photons',
                             verbose=True):
    """
    先复制GEDI文件，然后在副本中添加ICESat-2光子
    """

    print("\n" + "=" * 70)
    print("GEDI + ICESat-2 光子添加工具")
    print("=" * 70)

    # 1. 复制GEDI文件
    output_file = copy_gedi_file(original_gedi_file, output_suffix)
    if output_file is None:
        return None

    # 2. 加载ICESat-2光子数据
    print("\n[1/3] 加载ICESat-2光子数据...")
    with h5py.File(icesat2_file, 'r') as f:
        ph_lats = f['lat'][:]
        ph_lons = f['lon'][:]
        ph_h = f['h'][:]
        ph_conf = f['conf'][:] if 'conf' in f else np.ones(len(ph_lats), dtype=np.int8) * 2
        ph_is_signal = f['is_signal'][:] if 'is_signal' in f else (ph_conf >= 2)
        # 🔴 关键修改：只保留信号光子（is_signal=True）
        signal_mask = ph_is_signal
        print(f"    原始光子数: {len(ph_lats):,}")
        print(f"    信号光子数: {signal_mask.sum():,} ({100 * signal_mask.sum() / len(ph_lats):.1f}%)")

        ph_lats = ph_lats[signal_mask]
        ph_lons = ph_lons[signal_mask]
        ph_h = ph_h[signal_mask]
        ph_conf = ph_conf[signal_mask]
        ph_is_signal = ph_is_signal[signal_mask]  # 这里全部是 True
        if 'beam_source' in f:
            ph_beam_raw = f['beam_source'][:]
            ph_beam = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in ph_beam_raw]
        else:
            ph_beam = ['unknown'] * len(ph_lats)

        ph_dist = f['dist'][:] if 'dist' in f else np.zeros(len(ph_lats))

    print(f"    加载了 {len(ph_lats):,} 个光子")

    # 3. 建立光子网格索引
    print("\n[2/3] 建立光子空间索引...")
    grid_size = radius_deg * 2
    ph_grid = {}

    for i in tqdm(range(len(ph_lats)), desc="建立网格索引"):
        grid_lat = int(np.floor(ph_lats[i] / grid_size))
        grid_lon = int(np.floor(ph_lons[i] / grid_size))
        key = (grid_lat, grid_lon)
        if key not in ph_grid:
            ph_grid[key] = []
        ph_grid[key].append(i)

    print(f"    创建了 {len(ph_grid)} 个网格")

    # 4. 遍历副本文件，添加光子
    print(f"\n[3/3] 匹配并添加光子 (搜索半径={radius_deg * 111000:.0f}米)...")

    modified_count = 0
    total_photons_added = 0

    with h5py.File(output_file, 'r+') as gedi_f:
        beams = [k for k in gedi_f.keys() if k.startswith('BEAM')]
        print(f"    发现 {len(beams)} 个波束")

        for beam_name in tqdm(beams, desc="处理波束"):
            beam = gedi_f[beam_name]

            # 检查fpdata
            if 'fpdata' not in beam:
                if verbose:
                    print(f"      警告: {beam_name} 中没有 fpdata，跳过")
                continue

            fpdata = beam['fpdata']

            # 使用正确的字段名：lat_lowestmode, lon_lowestmode
            if 'lat_lowestmode' not in fpdata or 'lon_lowestmode' not in fpdata:
                if verbose:
                    print(f"      警告: {beam_name}/fpdata 中没有 lat_lowestmode/lon_lowestmode")
                    print(f"      可用字段: {list(fpdata.keys())}")
                continue

            # 读取所有足迹的经纬度（最低模式）
            fp_lats = fpdata['lat_lowestmode'][:]
            fp_lons = fpdata['lon_lowestmode'][:]
            pointnum = len(fp_lats)

            if verbose:
                print(f"\n      {beam_name}: {pointnum} 个足迹")
                print(
                    f"      经纬度范围: lat={fp_lats.min():.4f}~{fp_lats.max():.4f}, lon={fp_lons.min():.4f}~{fp_lons.max():.4f}")

            beam_modified = 0
            beam_photons = 0

            for i in range(pointnum):
                fp_lat = fp_lats[i]
                fp_lon = fp_lons[i]

                # 跳过无效坐标
                if np.isnan(fp_lat) or np.isnan(fp_lon):
                    continue

                # 计算所在网格
                grid_lat = int(np.floor(fp_lat / grid_size))
                grid_lon = int(np.floor(fp_lon / grid_size))

                # 搜索周围3x3网格
                nearby_indices = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        key = (grid_lat + di, grid_lon + dj)
                        if key in ph_grid:
                            nearby_indices.extend(ph_grid[key])

                if not nearby_indices:
                    continue

                # 精确计算距离
                distances = []
                valid_indices = []
                radius_m = radius_deg * 111000
                cos_lat = np.cos(np.radians(fp_lat))

                for idx in nearby_indices:
                    lat_diff = (ph_lats[idx] - fp_lat) * 111000
                    lon_diff = (ph_lons[idx] - fp_lon) * 111000 * cos_lat
                    dist = np.sqrt(lat_diff ** 2 + lon_diff ** 2)

                    if dist <= radius_m:
                        distances.append(dist)
                        valid_indices.append(idx)

                if not valid_indices:
                    continue

                # 创建足迹组
                fp_group_name = f'fp_{i:04d}'
                if fp_group_name not in beam:
                    fp_group = beam.create_group(fp_group_name)
                else:
                    fp_group = beam[fp_group_name]

                # 删除旧的photons组
                if 'photons' in fp_group:
                    del fp_group['photons']

                # 创建新的photons组
                ph_group = fp_group.create_group('photons')

                # 保存光子数据
                ph_group.create_dataset('h', data=ph_h[valid_indices], compression='gzip', compression_opts=6)
                ph_group.create_dataset('lat', data=ph_lats[valid_indices], compression='gzip', compression_opts=6)
                ph_group.create_dataset('lon', data=ph_lons[valid_indices], compression='gzip', compression_opts=6)
                ph_group.create_dataset('conf', data=ph_conf[valid_indices], compression='gzip', compression_opts=6)
                ph_group.create_dataset('is_signal', data=ph_is_signal[valid_indices], compression='gzip',
                                        compression_opts=6)
                ph_group.create_dataset('distance_m', data=np.array(distances, dtype=np.float32), compression='gzip',
                                        compression_opts=6)
                ph_group.create_dataset('beam', data=np.array(ph_beam)[valid_indices].astype('S'), compression='gzip',
                                        compression_opts=6)

                # 添加统计属性
                ph_group.attrs['n_photons'] = len(valid_indices)
                ph_group.attrs['n_signal'] = int(np.sum(ph_is_signal[valid_indices]))
                ph_group.attrs['mean_distance'] = float(np.mean(distances))
                ph_group.attrs['min_distance'] = float(np.min(distances))
                ph_group.attrs['max_distance'] = float(np.max(distances))
                ph_group.attrs['radius_m'] = radius_m

                beam_modified += 1
                beam_photons += len(valid_indices)

            if beam_modified > 0:
                modified_count += beam_modified
                total_photons_added += beam_photons
                if verbose:
                    print(f"      ✓ 添加了 {beam_modified} 个足迹的光子，共 {beam_photons} 个光子")

    # 5. 打印统计
    print("\n" + "=" * 70)
    print("添加完成统计")
    print("=" * 70)
    print(f"修改的足迹数: {modified_count}")
    print(f"添加的光子总数: {total_photons_added:,}")
    if modified_count > 0:
        print(f"平均每个足迹光子数: {total_photons_added / modified_count:.2f}")
    print(f"\n输出文件: {output_file}")

    return output_file


def inspect_photons_in_gedi(gedi_file, beam_name='BEAM0000', fp_index=0):
    """
    检查GEDI文件中的光子数据
    """
    print(f"\n检查文件: {gedi_file}")
    print("=" * 70)

    with h5py.File(gedi_file, 'r') as f:
        beams = [k for k in f.keys() if k.startswith('BEAM')]
        print(f"\n可用波束: {beams}")

        if beam_name not in f:
            print(f"波束 {beam_name} 不存在")
            return

        beam = f[beam_name]
        fp_group_name = f'fp_{fp_index:04d}'

        if fp_group_name not in beam:
            print(f"足迹 {fp_group_name} 不存在")
            fps = [k for k in beam.keys() if k.startswith('fp_')]
            if fps:
                print(f"可用足迹: {fps[:10]}...")
            return

        fp_group = beam[fp_group_name]

        print(f"\n波束: {beam_name}")
        print(f"足迹索引: {fp_index}")

        # 显示GEDI原始信息
        if 'fpdata' in beam:
            fpdata = beam['fpdata']
            if 'lat_lowestmode' in fpdata and fp_index < len(fpdata['lat_lowestmode']):
                print(f"\nGEDI信息:")
                print(f"  纬度(最低模式): {fpdata['lat_lowestmode'][fp_index]:.6f}")
                print(f"  经度(最低模式): {fpdata['lon_lowestmode'][fp_index]:.6f}")

                if 'elev_lowestmode' in fpdata:
                    print(f"  高程(最低模式): {fpdata['elev_lowestmode'][fp_index]:.2f}m")

        # 显示光子信息
        if 'photons' in fp_group:
            ph = fp_group['photons']
            print(f"\nICESat-2光子信息:")
            print(f"  光子数量: {ph.attrs['n_photons']}")
            print(f"  信号光子: {ph.attrs['n_signal']}")
            print(f"  距离范围: {ph.attrs['min_distance']:.2f} ~ {ph.attrs['max_distance']:.2f} m")
            print(f"  平均距离: {ph.attrs['mean_distance']:.2f} m")
            print(f"  搜索半径: {ph.attrs['radius_m']:.0f} m")

            if len(ph['h']) > 0:
                print(f"\n  前5个光子示例:")
                n_show = min(5, len(ph['h']))
                for i in range(n_show):
                    signal_str = "信号" if ph['is_signal'][i] else "噪声"
                    print(f"    光子{i}: h={ph['h'][i]:.2f}m, "
                          f"距离={ph['distance_m'][i]:.2f}m, "
                          f"置信度={ph['conf'][i]}, {signal_str}")
        else:
            print(f"\n该足迹没有光子数据")


# ==================== 主程序 ====================
if __name__ == "__main__":
    from datetime import datetime
    original_gedi_file = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5"
    icesat2_file = r"D:\研究生\SanFrancisco\ICESatdata\processed\icesat2_photons_20260422.h5"

    # 搜索半径：0.00025度 ≈ 28米 选成15m
    radius_deg = 0.000135
    # 获取当前日期，格式：YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")
    # 添加光子到GEDI副本
    output_file = add_photons_to_gedi_copy(
        original_gedi_file=original_gedi_file,
        icesat2_file=icesat2_file,
        radius_deg=radius_deg,
        output_suffix=f'_with_photons_15m{today}',
        verbose=True
    )

    # 检查结果
    if output_file and Path(output_file).exists():
        inspect_photons_in_gedi(output_file, beam_name='BEAM0000', fp_index=0)