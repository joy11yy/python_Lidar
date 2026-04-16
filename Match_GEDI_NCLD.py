import rasterio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from rasterio.mask import mask
from shapely.geometry import mapping,Point
import os
from Load_filtered_data import load_filtered_gedi_data
from Save_Match_Data import save_matched_data
from rasterio.warp import transform
import datetime
""""
rasterio 是处理TIF地理空间栅格数据，GEDI给的是经纬度lat/lon TIF存储的是行列号或者像素矩阵，rasterio
会读取TIF文件头里面的transfrom变换参数和crs坐标系，提供sample()或rowcol()函数，传入经纬度九年得到图片数值

"""



def match_gedi_ncld(gedi_filtered_file,lc_path):
    print("GEDI数据加载....")
    GEDIdata = load_filtered_gedi_data(gedi_filtered_file)

    if not GEDIdata:
        print("GEDI数据加载失败")
        return None

    try:
        with rasterio.open(lc_path) as src_lc:
            lc_crs = src_lc.crs
            lc_transform = src_lc.transform
            lc_nodata = src_lc.nodata
            height=src_lc.height
            width=src_lc.width

            print(f"栅格信息：crs={lc_crs},尺寸={src_lc.width}x{src_lc.height},nodata={lc_nodata}")
            lc_array = src_lc.read(1)
            print(f"   数组形状: {lc_array.shape}, 数据类型: {lc_array.dtype}")
            total_points = 0
            total_matched = 0
            #遍历每个波束进行匹配

            for beam_idx,beam_data in GEDIdata.items():
                beam_name=f"BEAM{beam_data['channel']}"
                n_points=beam_data['pointnum']
                total_points += n_points

                lons = np.array(beam_data['fpdata']['lon_lowestmode'], dtype=np.float64)
                lats = np.array(beam_data['fpdata']['lat_lowestmode'], dtype=np.float64)

                xs_proj, ys_proj = transform(
                    src_crs="EPSG:4326",
                    dst_crs=lc_crs,
                    xs=lons,
                    ys=lats
                )
                xs_proj = np.array(xs_proj)
                ys_proj = np.array(ys_proj)


                # xs, ys = lons, lats

                # 坐标行列号
                rows_list, cols_list = rowcol(lc_transform, xs_proj, ys_proj)
                rows = np.array(rows_list, dtype=np.int32)
                cols = np.array(cols_list, dtype=np.int32)

                land_cover_codes = np.full(n_points, -1, dtype=np.int16)
                in_bounds = (rows >= 0) & (rows < height)& (cols >= 0) & (cols < width)
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


            print(f"\n匹配完成！")
            print(f"   总点数：{total_points}")
            print(f"   成功匹配：{total_matched}")
            if total_points > 0:
                print(f"   成功率：{total_matched / total_points * 100:.2f}%")

                return GEDIdata
    except Exception as e:
        print(f"提取发生错误{e}")
        return None

#----------------------------------主程序入口-----------------------------------------
if __name__ == "__main__":
    NCLD_tif_path = r"D:\研究生\SanFrancisco\NCLD\NLCD_2024_SanFrancisco.tif"
    # GEDI_FILTERED_PATH = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
    #GEDI_FILTERED_PATH =r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2024361091112_O34205_02_T00048_02_006_02_V002.h5"
    #GEDI_FILTERED_PATH =r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_s0.5_2024361091112_O34205_02_T00048_02_006_02_V002.h5"
    # GEDI_FILTERED_PATH = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_1_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
    GEDI_FILTERED_PATH = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5"
    #OUTPUT_H5_PATH = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_matched_NLCD_4205_2024.h5"
    #获取当前日期
    current_date=datetime.datetime.now().strftime("%Y%m%d")
    OUTPUT_H5_PATH=rf"D:\研究生\SanFrancisco\GEDIdata\GEDI_matched_NLCD_{current_date}.h5"
    matched_data=match_gedi_ncld(GEDI_FILTERED_PATH,NCLD_tif_path)
    if matched_data is not None:
        save_matched_data(matched_data,OUTPUT_H5_PATH)
        print("\n 保存完毕")
    else:
        print("\n 匹配失败")

