import h5py
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from typing import Dict, Any, Optional
# 假设 Load_filtered_data 是你自定义的模块
import Load_filtered_data


def match_gedi_to_landcover(gedi_filtered_file: str, landcover_tif: str, output_file: Optional[str] = None):
    """
    将 GEDI 足迹点与地表覆盖 (Land Cover) TIF 数据进行空间匹配。
    增加了 CRS 检查和自动重投影功能。
    """

    print(f"   加载 GEDI 数据....")
    GEDIdata = Load_filtered_data.load_filtered_gedi_data(gedi_filtered_file)
    if not GEDIdata:
        print("GEDI 数据加载失败")
        return None

    # 1. 加载地表覆盖数据元数据
    print("\n[1/3] 读取地表覆盖 TIF 元数据...")
    with rasterio.open(landcover_tif) as src:
        lc_transform = src.transform
        lc_crs = src.crs
        nodata_val = src.nodata
        height, width = src.shape

        # 诊断信息
        print(f"TIF 尺寸：{width}*{height}")
        print(f"TIF CRS: {lc_crs}")
        print(f"边界：{src.bounds}")

        # === 关键修正：检查 CRS ===
        # GEDI 数据永远是 EPSG:4326 (WGS84)
        gedi_crs = "EPSG:4326"

        needs_reproject = False
        if lc_crs is None:
            print("⚠️ 警告：TIF 文件没有定义 CRS，假设为 EPSG:4326。如果实际是投影坐标，结果将错误！")
            # 强制假设，或者抛出错误，视严谨程度而定
            # raise ValueError("TIF missing CRS definition")
        elif str(lc_crs) != gedi_crs:
            print(f"⚠️ 检测到 CRS 不匹配 (TIF: {lc_crs} vs GEDI: {gedi_crs})，执行坐标重投影...")
            needs_reproject = True

        # 预读取波段数据 (注意：如果文件极大，这里可能会爆内存，需视具体情况优化)
        # 使用 masked array 可以更方便地处理 nodata，但为了性能先读普通数组
        lc_data = src.read(1)

    print("\n[2/3] 执行空间匹配...")
    total_matched = 0
    total_points = 0

    for beam_idx, beam_data in GEDIdata.items():
        n_points = beam_data['pointnum']
        total_points += n_points
        channel = beam_data.get('channel', beam_idx)
        print(f"处理波束 BEAM{channel} ({n_points} 个点)")

        # 获取经纬度 (确保是 numpy 数组)
        lons = np.array(beam_data['fpdata']['ins_lon'], dtype=np.float64)
        lats = np.array(beam_data['fpdata']['ins_lat'], dtype=np.float64)

        # === 核心修正：坐标重投影 ===
        if needs_reproject:
            try:
                # rasterio.warp.transform 可以将坐标从一个 CRS 转换到另一个
                # 输入：src_crs, dst_crs, xs, ys
                lons_proj, lats_proj = warp_transform(gedi_crs, lc_crs, lons, lats)
                lons_proj = np.array(lons_proj)
                lats_proj = np.array(lats_proj)
                xs, ys = lons_proj, lats_proj
            except Exception as e:
                print(f"❌ 坐标重投影失败：{e}")
                continue
        else:
            xs, ys = lons, lats

        # === 坐标转行列号 ===
        # rowcol 返回的是列表，可能包含浮点数（如果是精确位置），但用于索引时会被截断
        # 注意：rowcol 不会自动处理越界，越界的点返回的行列号可能是负数或超出范围
        rows_list, cols_list = rowcol(lc_transform, xs, ys)

        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)

        # 初始化结果数组 (-1 表示未匹配/无效)
        land_cover_codes = np.full(n_points, -1, dtype=np.int16)

        # === 边界检查 ===
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

        valid_count = np.sum(in_bounds)
        if valid_count == 0:
            print(f"         -> 无点在 TIF 范围内")
            continue

        # 提取有效索引
        valid_rows = rows[in_bounds]
        valid_cols = cols[in_bounds]
        valid_indices = np.where(in_bounds)[0]

        # 提取像素值
        extracted_codes = lc_data[valid_rows, valid_cols]

        # === 处理 NoData ===
        if nodata_val is not None:
            # 创建一个掩码，排除掉值为 nodata 的像素
            valid_data_mask = extracted_codes != nodata_val
            final_valid_indices = valid_indices[valid_data_mask]
            final_codes = extracted_codes[valid_data_mask]

            land_cover_codes[final_valid_indices] = final_codes
            matched_count = len(final_valid_indices)

            # 统计因 NoData 被过滤掉的点
            nodata_count = valid_count - matched_count
            if nodata_count > 0:
                print(
                    f"         -> 范围内 {valid_count} 点，其中 {nodata_count} 个为 NoData，成功匹配 {matched_count} 个")
        else:
            land_cover_codes[valid_indices] = extracted_codes
            matched_count = valid_count
            print(f"         -> 成功匹配 {matched_count} 个点")

        total_matched += matched_count
        beam_data['cover_type'] = land_cover_codes

    print(
        f"\n[3/3] 匹配完成。总点数：{total_points}, 成功匹配：{total_matched}, 成功率：{total_matched / total_points * 100:.2f}%")

    return GEDIdata