import numpy as np
import  os
import matplotlib.pyplot  as plt
def waveform_read(waveformdata, sample_start_index, sample_count, indices):
    """
    读取波形数据

    Parameters:
    -----------
    waveformdata : np.ndarray
        完整的波形数据
    sample_start_index : np.ndarray
        每个脉冲的起始索引
    sample_count : np.ndarray
        每个脉冲的采样点数
    indices : np.ndarray
        要提取的脉冲索引

    Returns:
    --------
    list
        包含每个脉冲波形的列表
    """
    waveforms = []

    for idx in indices:
        if idx <= len(sample_start_index):
            start = int(sample_start_index[idx - 1]) - 1  # MATLAB是1索引，Python是0索引
            count = int(sample_count[idx - 1])

            if start >= 0 and start + count <= len(waveformdata):
                waveform = waveformdata[start:start + count]
            else:
                waveform = np.array([])  # 无效索引返回空数组
        else:
            waveform = np.array([])

        waveforms.append(waveform)

    return waveforms


# 辅助函数：查看数据概览
def print_data_summary(GEDIdata):
    """
    打印GEDI数据的概览信息
    """
    for beam_idx, beam_data in GEDIdata.items():
        if beam_data['pointnum'] > 0:
            print(f"\n波束 {beam_data['channel']}:")
            print(f"  脉冲数量: {beam_data['pointnum']}")
            print(f"  有效数据: {beam_data['vailddata']}")

            # 显示第一个脉冲的部分信息
            if beam_data['wavedata']['txwaveform'] and len(beam_data['wavedata']['txwaveform']) > 0:
                tx_wf = beam_data['wavedata']['txwaveform'][0]
                rx_wf = beam_data['wavedata']['rxwaveform'][0]
                print(f"  发射波形长度: {len(tx_wf) if tx_wf is not None else 0}")
                print(f"  接收波形长度: {len(rx_wf) if rx_wf is not None else 0}")

            if beam_data['fpdata']:
                print(f"  第一个脉冲位置: ({beam_data['fpdata']['ins_lat'][0]:.6f}, "
                      f"{beam_data['fpdata']['ins_lon'][0]:.6f}, "
                      f"{beam_data['fpdata']['ins_alt'][0]:.2f}m)")


def draw_wave(GEDIdata,target_beam_code,pulse_index):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

    beam_data=None
    #data是每个通道里面包含的完整数据字典，items是字典的内置方法，返回字典中所有键值对的可迭代对象（键，值）

    for idx,data in GEDIdata.items():
        if data['channel']==target_beam_code:
            beam_data=data
            break
    if beam_data is None:
        print(f"未找到波束{target_beam_code}")
        return
    tx_waveforms=beam_data['wavedata'].get('txwaveform',[])
    rx_waveforms = beam_data['wavedata'].get('rxwaveform', [])

    if pulse_index<0 or pulse_index>=len(tx_waveforms):
        print(f"无效波束号，只含有{len(tx_waveforms)}个脉冲")
        return

    if len(tx_waveforms)==0 or len(rx_waveforms)==0:
        print(f"{target_beam_code}无波形数据")
        return

    tx_wf=tx_waveforms[pulse_index]
    rx_wf=rx_waveforms[pulse_index]

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8))
    fig.suptitle(f'GEDI_L1B {target_beam_code} 脉冲{pulse_index}')

    ax1.plot(tx_wf, color='#FF6B6B', linewidth=1.5, label='发射波形')
    ax1.set_title('发射波形 (Tx Waveform)', fontsize=12, fontweight='medium')
    ax1.set_ylabel('信号强度', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, len(tx_wf) - 1)

    ax2.plot(rx_wf, color='#4ECDC4', linewidth=1.5, label='接收波形')
    ax2.set_title('接收波形 (Rx Waveform)', fontsize=12, fontweight='medium')
    ax2.set_xlabel('采样点序号', fontsize=10)
    ax2.set_ylabel('信号强度', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, len(rx_wf) - 1)

    plt.tight_layout()
    plt.show()

