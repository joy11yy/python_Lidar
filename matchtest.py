import h5py
import numpy as np
import rasterio
from rasterio.transform import rowcol
import os

# ================= 配置区域 =================
# 请确保这两个路径是你刚才验证过的真实路径
GEDI_FILE = r"D:\研究生\SanFrancisco\GEDI_filtered_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
# 注意：如果你的 GEDI 文件还是那个全美国的原始文件，代码里的空间过滤步骤会帮你筛选出旧金山的点
# 如果你已经有一个只包含旧金山点的 GEDI 文件，请用那个文件的路径替换上面这行

TIF_FILE = r"D:\研究生\SanFrancisco\GLC_FCS30_2020_W125N40.tif"
OUTPUT_FILE = r"D:\研究生\SanFrancisco\GEDI_Matched_Result.h5"


# ===========================================

def load_and_match(gedi_path, tif_path, output_path):
    if not os.path.exists(gedi_path):
        print(f"❌ GEDI 文件不存在: {gedi_path}")
        return
    if not os.path.exists(tif_path):
        print(f"❌ TIF 文件不存在: {tif_path}")
        return

    print(f"📂 正在处理: {os.path.basename(gedi_path)}")

    # 1. 读取 TIF 信息
    with rasterio.open(tif_path) as src:
        transform = src.transform
        bounds = src.bounds
        lc_data = src.read(1)
        nodata = src.nodata
        height, width = lc_data.shape
        crs = src.crs

    print(f"✅ TIF 已加载: 范围 [{bounds.left:.2f}, {bounds.right:.2f}] x [{bounds.bottom:.2f}, {bounds.top:.2f}]")

    # 2. 读取 GEDI 数据并匹配
    total_points = 0
    matched_points = 0

    # 用于存储结果的字典
    results = {}

    try:
        with h5py.File(gedi_path, 'r') as f:
            # GEDI 数据结构通常是 BEAM0000, BEAM0001 等
            beams = [k for k in f.keys() if k.startswith('BEAM')]

            if not beams:
                # 尝试另一种常见的结构 (如果是经过预处理的文件)
                # 这里假设你的 Load_filtered_data 逻辑是把所有波束展平或者按特定结构存储
                # 为了通用性，我们直接遍历顶层 key，寻找包含 lat/lon 的数据集
                # 但根据你之前的输出，应该是有 BEAMxxxx 键的
                print("⚠️ 未找到标准的 BEAMxxxx 组，尝试查找其他结构...")
                # 如果文件结构不同，可能需要调整这里的遍历逻辑
                # 暂时假设结构与你之前展示的一致
                pass

            for beam_name in beams:
                beam_grp = f[beam_name]

                # 定位经纬度数据集 (根据你的文件结构调整路径，通常是 footprint 下的)
                # 常见路径: BEAM0000/footprint/latitude_bin0 或 BEAM0000/latitudes
                # 根据你之前的代码 Load_filtered_data，推测路径可能是:
                # beam_grp['fpdata']['ins_lat']
                # 让我们尝试直接访问标准 GEDI L2A 结构，或者你预处理后的结构

                # 假设结构：BEAMxxxx -> fpdata -> ins_lat / ins_lon
                if 'fpdata' in beam_grp:
                    fpdata = beam_grp['fpdata']
                    if 'ins_lat' in fpdata and 'ins_lon' in fpdata:
                        lats = fpdata['ins_lat'][:]
                        lons = fpdata['ins_lon'][:]
                    else:
                        continue
                else:
                    # 备用方案：直接找 lat/lon 数据集
                    continue

                n_pts = len(lats)
                total_points += n_pts

                # === 核心匹配逻辑 ===

                # 1. 快速边界过滤 (利用 Numpy 广播，速度极快)
                # 只保留在 TIF 范围内的点
                mask = (
                        (lons >= bounds.left) & (lons <= bounds.right) &
                        (lats >= bounds.bottom) & (lats <= bounds.top)
                )

                valid_lons = lons[mask]
                valid_lats = lats[mask]
                valid_indices = np.where(mask)[0]

                count_in_bounds = len(valid_lons)

                if count_in_bounds == 0:
                    print(f"   {beam_name}: {n_pts} 个点 -> 0 个在 TIF 范围内")
                    continue

                # 2. 计算行列号
                rows, cols = rowcol(transform, valid_lons, valid_lats)
                rows = np.array(rows).astype(int)
                cols = np.array(cols).astype(int)

                # 3. 二次检查 (防止 rowcol 边缘计算误差)
                valid_pixel_mask = (
                        (rows >= 0) & (rows < height) &
                        (cols >= 0) & (cols < width)
                )

                final_rows = rows[valid_pixel_mask]
                final_cols = cols[valid_pixel_mask]
                final_orig_indices = valid_indices[valid_pixel_mask]

                count_valid_pixels = len(final_rows)

                if count_valid_pixels == 0:
                    print(f"   {beam_name}: {count_in_bounds} 个在范围内 -> 0 个有效像素 (可能是 NoData 或边缘误差)")
                    continue

                # 4. 提取值
                values = lc_data[final_rows, final_cols]

                # 5. 过滤 NoData
                if nodata is not None:
                    data_mask = values != nodata
                    final_indices = final_orig_indices[data_mask]
                    final_values = values[data_mask]
                    count_final = len(final_indices)
                else:
                    final_indices = final_orig_indices
                    final_values = values
                    count_final = count_valid_pixels

                matched_points += count_final
                print(f"   {beam_name}: {n_pts} 总点 -> {count_in_bounds} 在范围内 -> 🎯 {count_final} 匹配成功")

                # 保存结果 (可选：将匹配到的值存回字典，稍后写入文件)
                # 这里为了简单，先打印统计。如果需要保存新文件，可以在此处构建数组
                results[beam_name] = {
                    "indices": final_indices,
                    "values": final_values
                }

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 50)
    print(f"📊 最终统计:")
    print(f"   GEDI 总点数: {total_points}")
    print(f"   成功匹配点数: {matched_points}")
    print(f"   匹配率: {matched_points / total_points * 100:.2f}%")
    print("=" * 50)

    if matched_points > 1000:
        print("🎉 成功！现在你有足够的数据进行分析了！")
    else:
        print("⚠️ 匹配点数依然较少，请检查 GEDI 文件是否真的包含旧金山区域的高质量数据。")


if __name__ == "__main__":
    load_and_match(GEDI_FILE, TIF_FILE, OUTPUT_FILE)