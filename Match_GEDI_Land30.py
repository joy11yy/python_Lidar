import os

import h5py
from  rasterio.merge import  merge
import numpy as np
import rasterio
from rasterio import vrt as rio_vrt
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from typing import Dict, Any, Optional
import Load_filtered_data
from typing import  List
from rasterio.warp import transform
from FeatureExtract import extract_waveform_features
from Save_Match_Data import save_matched_data


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
    total_matched = 0k
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


def merge_tifs_to_array(tif_list,nodata_value=None):
    """
      将多个 TIF 文件拼接成一个大的 NumPy 数组。
      返回: (data_array, transform, crs, nodata)
      """
    if not tif_list:
        raise ValueError("TIF 列表为空")

    print(f"🔍 正在检查并打开 {len(tif_list)} 个文件...")
    datasets = []
    try:
        # 1. 打开所有文件
        for path in tif_list:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            ds = rasterio.open(path)
            datasets.append(ds)
            print(f"   - 已加载: {os.path.basename(path)} ({ds.width}x{ds.height})")

        # 2. 执行拼接 (Merge)
        # method='first' 表示重叠区域取第一个文件的值（对于分类数据通常够用）
        # nodata=nodata_value 指定背景值
        print("🔧 正在执行像素级拼接 (Merge)...")
        mosaic, out_transform = merge(datasets, method='first', nodata=nodata_value)

        # 3. 获取元数据
        # 假设所有输入文件的 CRS 和 dtype 是一致的，取第一个文件的属性
        out_crs = datasets[0].crs
        out_nodata = datasets[0].nodata if datasets[0].nodata is not None else nodata_value

        print(f"✅ 拼接完成！最终尺寸：{mosaic.shape[2]} (宽) x {mosaic.shape[1]} (高)")
        print(f"   CRS: {out_crs}")
        print(f"   Transform: {out_transform}")

        # 返回数组 (只取第一个波段，因为土地覆盖通常是单波段)
        return mosaic[0], out_transform, out_crs, out_nodata

    finally:
        # 4. 关闭所有文件句柄 (非常重要，防止文件占用)
        for ds in datasets:
            ds.close()
        print("🔒 已关闭所有文件句柄。")

#CRS 是Coordinate Reference System 坐标参考系统 EPSG:4326国际代号代表WGS84坐标系
def match_gedi_to_landcover_multi(gedi_filtered_file: str, landcover_tif_list: List[str], output_file: Optional[str] = None, nodata_value: Optional[float] = None):
    print("GEDI数据加载....")
    GEDIdata=Load_filtered_data.load_filtered_gedi_data(gedi_filtered_file)
    if not GEDIdata:
        print("GEDI数据加载失败")
        return None

    datasets=[]
    try:
        lc_array, lc_transform, lc_crs, lc_nodata = merge_tifs_to_array(landcover_tif_list, nodata_value=nodata_value)

    except Exception as e:
        print(f"拼接失败")
        return None
    # === CRS 检查 ===
    target_crs = "EPSG:4326"
    needs_reproject = False
    if str(lc_crs) != target_crs:
        print(f"\n⚠️ 警告：土地覆盖数据 CRS ({lc_crs}) 与 GEDI ({target_crs}) 不一致！")
        print("   当前简易拼接模式不支持自动重投影数组。")
        print("   如果数据不是 EPSG:4326，匹配结果将会有偏差。")
        print("   (若需精确重投影，建议使用 GDAL VRT 方案)")

    else:
        print(f"\n✅ CRS 检查通过：{lc_crs} (无需重投影)")

    print("执行空间匹配")
    height, width = lc_array.shape
    total_matched = 0
    total_points = 0

    for beam_idx, beam_data in GEDIdata.items():
        n_points = beam_data['pointnum']
        total_points += n_points
        channel = beam_data.get('channel', beam_idx)
        print(f"处理波束 BEAM{channel} ({n_points} 个点)")
        lons=np.array(beam_data['fpdata']['lon_lowestmode'], dtype=np.float64)
        lats=np.array(beam_data['fpdata']['lat_lowestmode'], dtype=np.float64)

        xs_proj, ys_proj = transform(
            src_crs="EPSG:4326",
            dst_crs=lc_crs,
            xs=lons,
            ys=lats
        )
        xs_proj = np.array(xs_proj)
        ys_proj = np.array(ys_proj)

        # xs, ys = lons, lats


        #坐标行列号
        rows_list, cols_list = rowcol(lc_transform, xs_proj,ys_proj)
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)

        land_cover_codes = np.full(n_points, -1, dtype=np.int16)
        in_bounds = (rows >= 0) & (rows < height)
        valid_count = np.sum(in_bounds)
        if valid_count == 0:
            continue
        valid_rows = rows[in_bounds]
        valid_cols = cols[in_bounds]
        valid_indices = np.where(in_bounds)[0]
        extracted_codes = lc_array[valid_rows, valid_cols]
        # 处理 NoData
        if lc_nodata is not None:
            valid_data_mask = extracted_codes != lc_nodata
            final_valid_indices = valid_indices[valid_data_mask]
            final_codes = extracted_codes[valid_data_mask]
            land_cover_codes[final_valid_indices] = final_codes
            matched_count = len(final_valid_indices)
        else:
            land_cover_codes[valid_indices] = extracted_codes
            matched_count = valid_count

        total_matched += matched_count
        beam_data['cover_type'] = land_cover_codes

        # 可选：打印进度
        # if beam_idx % 2 == 0:
        #     print(f"   处理波束 {beam_idx} ...")

    print(f"\n匹配完成！")
    print(f"   总点数：{total_points}")
    print(f"   成功匹配：{total_matched}")
    if total_points > 0:
        print(f"   成功率：{total_matched / total_points * 100:.2f}%")

    return GEDIdata


#----------------------------------主程序入口-----------------------------------------
if __name__ == "__main__":
    # GEDI_FILTERED_PATH = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
    gedi_filtered_file=r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2025009102237_O34423_03_T04153_02_006_02_V002.h5"

    output_file = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_matched_GLC30.h5"

    #多个TIF匹配
    landcover_tif_list=[r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N25.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N30.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N25.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W125N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W125N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N30.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N30.tif"

    ]
    GEDIdata_match=match_gedi_to_landcover_multi(gedi_filtered_file,landcover_tif_list,output_file)

    print(f"\n[2/2] 正在保存精简版数据到: {output_file}")
    try:
        saved_path=save_matched_data(
            GEDIdata_match,
            output_file)
        print("保存成功")
    except Exception as e:
        print(f"保存失败：{e}")
        import traceback
        traceback.print_exc()
