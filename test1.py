import h5py
import numpy as np
import os

# ================= 配置区域 =================
GEDI_FILE = r"D:\研究生\SanFrancisco\GEDI_filtered_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
# TIF 范围 (硬编码，确保和刚才诊断的一致)
TIF_LEFT, TIF_RIGHT = -125.0, -120.0
TIF_BOTTOM, TIF_TOP = 35.0, 40.0
# ===========================================

if not os.path.exists(GEDI_FILE):
    print(f"❌ 文件不存在: {GEDI_FILE}")
else:
    print(f"🔍 正在深度检查 GEDI 文件内部坐标...\n")
    print(f"🎯 目标范围 (TIF): Lon [{TIF_LEFT}, {TIF_RIGHT}], Lat [{TIF_BOTTOM}, {TIF_TOP}]")
    print("-" * 80)

    with h5py.File(GEDI_FILE, 'r') as f:
        beams = [k for k in f.keys() if k.startswith('BEAM')]

        total_checked = 0
        total_in_range = 0

        for beam_name in beams:
            beam_grp = f[beam_name]

            # 尝试寻找经纬度路径
            lats = None
            lons = None

            # 路径尝试 1: fpdata/ins_lat (你之前的代码用的)
            if 'fpdata' in beam_grp and 'ins_lat' in beam_grp['fpdata']:
                lats = beam_grp['fpdata']['ins_lat'][:]
                lons = beam_grp['fpdata']['ins_lon'][:]

            # 路径尝试 2: 直接在 beam 下 (有些处理后的文件)
            elif 'latitude' in beam_grp and 'longitude' in beam_grp:
                lats = beam_grp['latitude'][:]
                lons = beam_grp['longitude'][:]

            # 路径尝试 3: geolocation 组
            elif 'geolocation' in beam_grp:
                if 'latitude_bin0' in beam_grp['geolocation']:  # GEDI L2A 标准结构
                    lats = beam_grp['geolocation']['latitude_bin0'][:]
                    lons = beam_grp['geolocation']['longitude_bin0'][:]

            if lats is None:
                print(f"⚠️ {beam_name}: 未找到经纬度数据集，跳过。")
                continue

            n_pts = len(lats)
            if n_pts == 0:
                continue

            # 取前 10 个点进行“显微镜”检查
            sample_size = min(10, n_pts)
            sample_lats = lats[:sample_size]
            sample_lons = lons[:sample_size]

            # 检查这 10 个点有多少在范围内
            in_range_mask = (
                    (sample_lons >= TIF_LEFT) & (sample_lons <= TIF_RIGHT) &
                    (sample_lats >= TIF_BOTTOM) & (sample_lats <= TIF_TOP)
            )
            count_in_sample = np.sum(in_range_mask)

            # 检查整个波束有多少在范围内 (向量化操作，很快)
            full_mask = (
                    (lons >= TIF_LEFT) & (lons <= TIF_RIGHT) &
                    (lats >= TIF_BOTTOM) & (lats <= TIF_TOP)
            )
            count_full = np.sum(full_mask)

            total_checked += n_pts
            total_in_range += count_full

            print(f"\n📡 波束: {beam_name} (总点数: {n_pts})")
            print(f"   📊 整个波束落在 TIF 范围内的点数: {count_full}")
            print(f"   🔬 前 {sample_size} 个点抽样检查:")

            for i in range(sample_size):
                lon = sample_lons[i]
                lat = sample_lats[i]
                status = "✅ 在范围内" if in_range_mask[i] else "❌ 越界"
                # 格式化输出，保留 6 位小数，方便看细微差别
                print(f"      [{i}] Lon: {lon:>10.6f}, Lat: {lat:>10.6f} -> {status}")

            # 如果前 10 个都不在，但总数说有在，那说明数据分布很奇怪，需要警惕
            if count_full > 0 and count_in_sample == 0:
                print(f"   ⚠️ 警告: 前 10 个点都不在范围内，但整个波束有 {count_full} 个点在范围内。数据分布可能不均匀。")

    print("\n" + "=" * 80)
    print(f"📝 总结:")
    print(f"   检查总点数: {total_checked}")
    print(f"   落在 TIF 范围内的总点数: {total_in_range}")

    if total_in_range == 0:
        print("\n❌ 确诊: GEDI 文件中确实没有任何点落在 [-125, -120] x [35, 40] 区域内！")
        print("   可能原因:")
        print("   1. 这个 H5 文件虽然是 'filtered'，但过滤条件可能把旧金山的点都剔除了。")
        print("   2. 文件里的经度数据可能是错误的 (例如全是 0，或者变成了正数 125 而不是 -125)。")
        print("   3. 文件名里的轨道号 (O34785) 对应的实际轨迹可能根本不经过旧金山，尽管统计范围看起来像。")
    else:
        print(f"\n🎉 找到了 {total_in_range} 个点！之前的匹配代码逻辑可能有误，或者数据类型有问题。")