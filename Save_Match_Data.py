import h5py
import numpy as np
import os
from datetime import datetime


def save_matched_data(GEDIdata, output_file, landcover_source=None):
    """
    保存【匹配后且已剔除无效点】的 GEDI 数据。

    逻辑：
    1. 遍历每个波束。
    2. 找到 cover_type != -1 的有效索引。
    3. 利用这些索引，对 shot_number, deltatime, fpdata, wavedata, cover_type 等所有数组进行同步切片。
    4. 只保存切片后的数据。

    参数:
    GEDIdata: 包含 'cover_type' 字段的字典 (其中 -1 代表无效)
    output_file: 输出文件路径
    landcover_source: (可选) 记录使用的 TIF 文件名
    """
    print(f"\n 正在保存匹配数据到: {output_file}")
    print("   (仅保留 cover_type != -1 的有效点)")

    total_original = 0
    total_saved = 0

    with h5py.File(output_file, 'w') as f:
        # 添加元数据
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['processing_type'] = 'Filtered_Matched_Compact'  # 标记为精简版
        f.attrs['description'] = 'Only shots with valid land cover codes are saved.'
        if landcover_source:
            f.attrs['landcover_source'] = landcover_source

        # 遍历每个波束进行筛选和保存
        for beam_idx, beam_data in GEDIdata.items():
            beam_name = f"BEAM{beam_data['channel']}"
            n_original = beam_data['pointnum']
            total_original += n_original

            # === 核心步骤 1: 生成有效掩膜 ===
            if 'cover_type' not in beam_data:
                print(f"   ⚠️ 警告: 波束 {beam_name} 缺少 'cover_type' 字段，跳过该波束。")
                continue

            codes = beam_data['cover_type']
            valid_mask = (codes != -1)  # True 表示有效，False 表示无效

            n_valid = np.sum(valid_mask)
            total_saved += n_valid

            if n_valid == 0:
                print(f"   波束 {beam_name}: 无有效匹配点，跳过保存。")
                continue

            print(f"   处理波束 {beam_name}: {n_original} -> {n_valid} 个点 (剔除 {n_original - n_valid})")

            # === 核心步骤 2: 创建波束组并保存筛选后的数据 ===
            beam_group = f.create_group(beam_name)
            beam_group.attrs['pointnum'] = n_valid  # 更新点数为有效点数
            beam_group.attrs['original_pointnum'] = n_original  # 记录原始点数以便追溯

            # 辅助函数：保存普通数组 (应用掩膜)
            def save_array(group, name, data):
                group.create_dataset(name, data=data[valid_mask], compression='gzip')

            # 1. 保存基础数据
            save_array(beam_group, 'shot_number', beam_data['shot_number'])
            save_array(beam_group, 'delta_time', beam_data['deltatime'])

            # 2. 保存质量信息 (如果存在)
            if 'quality' in beam_data:
                quality_group = beam_group.create_group('quality')
                for key, value in beam_data['quality'].items():
                    save_array(quality_group, key, value)

            # 3. 保存波形数据 (最复杂的部分，因为包含变长数组)
            if 'wavedata' in beam_data:
                wavedata_group = beam_group.create_group('wavedata')

                # 处理变长波形 (txwaveform, rxwaveform)
                for wave_key in ['txwaveform', 'rxwaveform']:
                    if wave_key in beam_data['wavedata']:
                        all_waveforms = beam_data['wavedata'][wave_key]
                        # 列表推导式：只保留 mask 为 True 的波形
                        filtered_waveforms = [w for i, w in enumerate(all_waveforms) if valid_mask[i]]

                        dt = h5py.vlen_dtype(np.dtype('float32'))
                        wave_dataset = wavedata_group.create_dataset(wave_key, (len(filtered_waveforms),), dtype=dt,
                                                                     compression='gzip')
                        for i, w in enumerate(filtered_waveforms):
                            wave_dataset[i] = np.array(w, dtype=np.float32)

                # 处理普通波形数组 (rx_sample_count, rx_energy)
                for key in ['rx_sample_count', 'rx_energy']:
                    if key in beam_data['wavedata']:
                        save_array(wavedata_group, key, beam_data['wavedata'][key])

            # 4. 保存定位数据
            if 'fpdata' in beam_data:
                fpdata_group = beam_group.create_group('fpdata')
                for key, value in beam_data['fpdata'].items():
                    save_array(fpdata_group, key, value)

            # 5. 保存噪声数据
            if 'noisedata' in beam_data:
                noisedata_group = beam_group.create_group('noisedata')
                for key, value in beam_data['noisedata'].items():
                    save_array(noisedata_group, key, value)

            # 6. 保存核心的地类代码 (此时已经全是有效值了，不再有 -1)
            beam_group.create_dataset(
                'cover_type',
                data=codes[valid_mask],
                dtype='int16',
                compression='gzip'
            )

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    reduction_rate = (1 - total_saved / total_original) * 100 if total_original > 0 else 0

    print(f"\n✅ 保存完成！")
    print(f"   原始总点数: {total_original}")
    print(f"   保存有效点数: {total_saved}")
    print(f"   数据压缩率: {reduction_rate:.1f}% (剔除了 {total_original - total_saved} 个无效点)")
    print(f"   文件大小: {file_size:.2f} MB")

    return output_file