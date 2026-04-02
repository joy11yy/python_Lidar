import h5py
import pandas as pd
import os
import numpy as np

# === 配置区域 ===
# 这里填入你刚才生成的 filtered 文件路径
# 根据你之前的代码，它应该在同目录下，文件名类似 GEDI_filtered_...h5
# filtered_file_path = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2025009102237_O34423_03_T04153_02_006_02_V002.h5"
filtered_file_path = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2024361091112_O34205_02_T00048_02_006_02_V002.h5"
    #GEDI_FILTERED_PATH = r

# 如果上面的文件不存在，尝试使用原始的 L2A 文件路径作为备选
# original_l2a = r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_2025009102237_O34423_03_T04153_02_004_02_V002.h5"

def extract_coords_to_csv(h5_file_path, output_csv):
    print(f"📂 正在读取文件: {h5_file_path}")

    if not os.path.exists(h5_file_path):
        print(f"❌ 错误：文件不存在！请检查路径: {h5_file_path}")
        return

    all_lats = []
    all_lons = []
    all_beams = []
    all_shot_numbers = []

    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 遍历文件中的所有组 (BEAM0000, BEAM0001, ...)
            for beam_name in f.keys():
                if beam_name.startswith('BEAM'):
                    print(f"  📡 处理波束: {beam_name}")
                    beam_group = f[beam_name]

                    # GEDI L2A 的标准经纬度路径通常在 fpdata 组下
                    # 键名通常是 'latitude_bin0' 和 'longitude_bin0'
                    fpdata_group = beam_group['fpdata']

                    lat_key = 'lat_lowestmode'
                    lon_key = 'lon_lowestmode'

                    if lat_key in fpdata_group and lon_key in fpdata_group:
                        lats = fpdata_group[lat_key][:]
                        lons = fpdata_group[lon_key][:]

                        # 获取 shot_number 用于标识 (如果在根目录或 quality 里)
                        shots = beam_group['shot_number'][:] if 'shot_number' in beam_group else np.arange(len(lats))

                        all_lats.extend(lats)
                        all_lons.extend(lons)
                        all_beams.extend([beam_name] * len(lats))
                        all_shot_numbers.extend(shots)

                        print(f"     提取了 {len(lats)} 个点")
                    else:
                        print(f"     ⚠️ 警告: 在 {beam_name} 中未找到 {lat_key} 或 {lon_key}")
                        print(f"     可用键值: {list(fpdata_group.keys())}")

        if len(all_lats) == 0:
            print("❌ 没有提取到任何坐标数据！")
            return

        # 创建 DataFrame
        df = pd.DataFrame({
            'longitude': all_lons,
            'latitude': all_lats,
            'beam': all_beams,
            'shot_number': all_shot_numbers
        })

        # 保存 CSV
        df.to_csv(output_csv, index=False)

        print("\n" + "=" * 40)
        print(f"✅ 成功！坐标已提取并保存。")
        print(f"📄 输出文件: {output_csv}")
        print(f"📊 总点数: {len(df)}")
        print(f"🗺️ 经纬度范围:")
        print(f"   Lat: {df['latitude'].min():.4f} ~ {df['latitude'].max():.4f}")
        print(f"   Lon: {df['longitude'].min():.4f} ~ {df['longitude'].max():.4f}")
        print("=" * 40)
        print("\n👉 下一步操作:")
        print(f"   1. 打开 Google Earth Engine (code.earthengine.google.com)")
        print(f"   2. 点击左侧 'Assets' -> '+ NEW' -> 'File upload'")
        print(f"   3. 选择文件: {output_csv}")
        print(f"   4. 等待上传完成后，复制资产路径 (users/xxx/...)")
        print(f"   5. 运行我下一条回复提供的 GEE 验证代码。")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


# 定义输出 CSV 路径 (保存在同目录)
output_dir = os.path.dirname(filtered_file_path)
output_csv_path = os.path.join(output_dir, "gedi_sf_points_for_gee4205.csv")

# 执行提取
extract_coords_to_csv(filtered_file_path, output_csv_path)