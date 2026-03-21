import numpy as np
from scipy.optimize import lsq_linear, least_squares

import matplotlib.pyplot as plt


def waveresolve(rx, ry, filtwidth, signalextent=None, noise_sigma=1, txsigma=1, maxwavenum=6, display=0):
    """
    基于拐点法检测的波形分解方法
    基础版本：主要基于 Decomposition of Laser Altimeter Waveforms 文献
    （1） 增添信号范围识别，仅考虑信号范围之外的数据；
    （2） 不考虑拐点对之间最大幅值小于噪声阈值的数据；

    参数说明：
    --------
    rx: np.ndarray
        x轴采样点（对应MATLAB的rx）
    ry: np.ndarray
        原始接收波形（对应MATLAB的ry）
    filtwidth: float
        滤波宽度
    signalextent: list/tuple, optional
        信号范围 [sigbgn, sigend]，默认None自动识别
    noise_sigma: float
        噪声标准差，默认1
    txsigma: float
        发射波形sigma，默认1
    maxwavenum: int
        最大高斯分量数，默认6
    display: int
        是否显示拟合结果图（1显示，0不显示）

    返回：
    ----
    prfnl: np.ndarray
        最终拟合的高斯参数（每行：幅值, 中心位置, 脉宽）
    prini: np.ndarray
        初始估计的高斯参数
    """

 # 生成高斯滤波核
    txwave = gaussfuction(filtwidth, 6, 1)
    # 卷积滤波（same模式，保持长度一致）
    rxwave_conv = np.convolve(ry, txwave, mode='same')
    signal = rxwave_conv
    wavelen = len(signal)

    # # 背景噪声阈值
    # Thres = noise_sigma * 3  # 阈值
    # minamp = Thres  # 最小振幅"""
    #自适应阈值
    signal_max=np.max(signal)
    base_threshold=noise_sigma*3
    if signal_max<base_threshold:
        adaptive_threshold = max(signal_max * 0.8, noise_sigma * 2)
        print(f"信号较弱 (最大值={signal_max:.2f} < 3σ={base_threshold:.2f})")
        print(f"使用自适应阈值: {adaptive_threshold:.2f}")
        Thres = adaptive_threshold
    else:
        Thres=base_threshold
        print(f"信号较强，使用3σ阈值: {Thres:.2f}")

    minamp = Thres  # 最小振幅

    # ========== 寻找信号范围 ==========
    if signalextent is not None and len(signalextent) == 2:
        sigbgn = signalextent[0]
        sigend = signalextent[1]
    else:
        # 找出所有大于阈值的点
        index = np.where(signal > Thres)[0]
        if len(index) == 0:
            sigbgn, sigend = 2, wavelen - 1
        else:
            sigbgn = max(index[0] - 50, 2)
            sigend = min(index[-1] + 50, wavelen - 1)

    # ========== 拐点检测 ==========
    # 计算一阶和二阶导数
    dif1 = np.zeros(wavelen)
    dif2 = np.zeros(wavelen)
    dif1[1:] = signal[1:] - signal[:-1]
    dif2[1:] = dif1[1:] - dif1[:-1]

    # 替换0值防止连续拐点
    dif2[dif2 == 0] = 1

    # 存储拐点：inf[0,:]位置，inf[1,:]类型(1上升，2下降)
    inf = np.zeros((2, wavelen))
    j = 0  # Python从0索引

    # 起始为下降沿则跳过第一个拐点
    if dif2[1] < 0:
        j += 1

    # 遍历有效信号范围检测拐点
    for i in range(int(sigbgn), int(sigend) - 1):
        if dif2[i] * dif2[i + 1] < 0:  # 相邻二阶导数符号相反，存在拐点
            if dif2[i] > 0:  # 上升拐点
                inf[0, j] = i
                inf[1, j] = 1
                j += 1
            elif dif2[i] < 0:  # 下降拐点
                inf[0, j] = i
                inf[1, j] = 2
                j += 1

    # 剔除0值列
    loc = inf[0, :] == 0
    inf = inf[:, ~loc]

    # ========== 提取拐点对并计算最大幅值 ==========
    inf1 = inf[0, :]  # 拐点位置
    inf2 = inf[1, :]  # 拐点类型
    prmtx1 = []

    while len(inf1) > 0:
        if inf2[0] == 1:  # 起始为上升沿
            # 找第一个下降沿
            matchdown = np.where(inf2 == 2)[0]
            if len(matchdown) == 0:
                break
            matchdown = matchdown[0]

            # 计算该拐点对之间的最大幅值
            start_idx = int(inf1[0])
            end_idx = int(inf1[matchdown])
            maxamp = np.max(signal[start_idx:end_idx + 1])

            if maxamp >= minamp:
                prmtx1.append([maxamp, inf1[0], inf1[matchdown]])

            # 删除已处理的拐点对
            inf1 = np.delete(inf1, [0, matchdown])
            inf2 = np.delete(inf2, [0, matchdown])
        else:
            # 起始为下降沿，直接删除
            inf1 = np.delete(inf1, 0)
            inf2 = np.delete(inf2, 0)

    prmtx1 = np.array(prmtx1) if prmtx1 else np.zeros((0, 3))

    # ========== 高斯参数初始估计 ==========
    # 按幅值升序排序
    if len(prmtx1) > 0:
        prmtx1 = prmtx1[np.argsort(prmtx1[:, 0])]
    prini = prmtx1.copy()

    # 计算初始中心位置和脉宽
    if len(prini) > 0:
        prini[:, 1] = (prmtx1[:, 1] + prmtx1[:, 2]) / 2  # 中心位置
        prini[:, 2] = (prmtx1[:, 2] - prmtx1[:, 1]) / 2  # 脉宽

    # 剔除无效数据
    if len(prini) > 0:
        loc = prini[:, 2] <= 1
        prini = prini[~loc]
        prini = prini[prini[:, 1] >= minamp]

    # 限制高斯分量数量
    gnum = maxwavenum
    if len(prini) > gnum:
        # 按面积排序（幅值*脉宽）
        area = prini[:, 0] * prini[:, 2]
        prini = np.hstack([area.reshape(-1, 1), prini])
        # 按面积降序排序
        prini = prini[np.argsort(prini[:, 0])[::-1]]
        # 保留前gnum个
        prini = prini[:gnum, 1:]

    # ========== 非线性最小二乘拟合 ==========
    # 设置优化选项
    options = {
        'method': 'trf',  # 替代trust-region-reflective
        'verbose': 0,  # 静默模式
        'max_nfev': 1000
    }

    if len(prini) == 0:
        return np.array([]), np.array([])

    # 设置上下限
    lb = prini.copy()
    lb[:, 0] = 0  # 幅值下限
    lb[:, 1] = sigbgn  # 位置下限
    lb[:, 2] = txsigma  # 脉宽下限

    ub = prini.copy()
    ub[:, 0] = np.max(ry)  # 幅值上限
    ub[:, 1] = sigend  # 位置上限
    ub[:, 2] = np.inf  # 脉宽上限

    # 展平参数用于优化
    x0 = prini.flatten()

    # 定义拟合目标函数
    def fit_func(x):
        return fit_gauss12(x.reshape(-1, 3), ry, [sigbgn, sigend])

    # 执行最小二乘拟合
    try:
        result = least_squares(
            fit_func,
            x0,
            bounds=(lb.flatten(), ub.flatten()),
            method="trf",
            max_nfev=1000,
            verbose=0,
        )
        prfnl = result.x.reshape(-1, 3)
    except:
        prfnl = prini.copy()

    # 按位置排序
    if len(prfnl) > 0:
        prfnl = prfnl[np.argsort(prfnl[:, 1])]

    # 剔除小幅值分量并合并邻近分量
    loc = prfnl[:, 0] < minamp if len(prfnl) > 0 else []
    d_loc = np.sum(np.diff(prfnl[:, 1]) < 6.6) > 0 if len(prfnl) > 1 else False

    while np.sum(loc) > 0 or d_loc:
        # 剔除小于阈值的分量
        if len(prfnl) > 0:
            prfnl = prfnl[~loc]
        # 合并邻近分量
        prfnl = wave_merged(prfnl, 15.6 / 2.355)

        if len(prfnl) == 0:
            break

        # 重新设置上下限
        lb = prfnl.copy()
        lb[:, 0] = 0
        lb[:, 1] = sigbgn
        lb[:, 2] = txsigma

        ub = prfnl.copy()
        ub[:, 0] = np.max(ry)
        ub[:, 1] = sigend
        ub[:, 2] = np.inf

        # 重新拟合
        try:
            result = least_squares(
                fit_func,
                prfnl.flatten(),
                bounds=(lb.flatten(), ub.flatten()),
                method="trf",
                max_nfev=1000,
                verbose=0,
            )
            prfnl = result.x.reshape(-1, 3)
        except:
            break

        # 更新判断条件
        loc = prfnl[:, 0] < minamp if len(prfnl) > 0 else []
        d_loc = np.sum(np.diff(prfnl[:, 1]) < 6.6) > 0 if len(prfnl) > 1 else False

    # ========== 绘图显示 ==========
    if display == 1 and len(prfnl) > 0:
        # 计算拟合的高斯波形
        gss80s = np.zeros((len(prfnl), wavelen))
        x = np.arange(1, wavelen + 1)
        for i in range(len(prfnl)):
            amp, pos, sigma = prfnl[i]
            gss80s[i, :] = amp * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))
        gss80 = np.sum(gss80s, axis=0)

        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(ry, label='原始波形', color='k')
        plt.plot(gss80, 'b--', linewidth=2, label='拟合总波形')
        for i in range(len(gss80s)):
            plt.plot(gss80s[i, :], 'm', alpha=0.6, label=f'高斯分量{i + 1}')

        plt.xlim(0, len(ry))
        plt.ylim(0, np.max(ry) * 1.1)
        plt.xlabel('采样点')
        plt.ylabel('信号强度')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return prfnl, prini


def wave_merged(prfnl, sigma):
    """
    对间隔小于发射波形间隔的波形进行合并

    参数：
    ----
    prfnl: np.ndarray
        高斯参数数组（幅值, 中心位置, 脉宽）
    sigma: float
        合并阈值

    返回：
    ----
    prfnl: np.ndarray
        合并后的高斯参数
    """
    if len(prfnl) <= 1:
        return prfnl

    # 按位置排序
    prfnl = prfnl[np.argsort(prfnl[:, 1])]
    waveindex = 0

    while waveindex < len(prfnl) - 1:
        # 检查相邻分量的位置间隔
        if prfnl[waveindex + 1, 1] - prfnl[waveindex, 1] < sigma:
            # 计算面积权重
            area1 = prfnl[waveindex, 0] * prfnl[waveindex, 2]
            area2 = prfnl[waveindex + 1, 0] * prfnl[waveindex + 1, 2]
            total_area = area1 + area2
            w1 = area1 / total_area if total_area > 0 else 0.5
            w2 = 1 - w1

            # 合并参数
            prfnl[waveindex, 0] = max(prfnl[waveindex, 0], prfnl[waveindex + 1, 0])  # 幅值取最大
            prfnl[waveindex, 1] = w1 * prfnl[waveindex, 1] + w2 * prfnl[waveindex + 1, 1]  # 加权中心
            prfnl[waveindex, 2] = w1 * prfnl[waveindex, 2] + w2 * prfnl[waveindex + 1, 2]  # 加权脉宽

            # 删除被合并的分量
            prfnl = np.delete(prfnl, waveindex + 1, axis=0)
        else:
            waveindex += 1

    return prfnl


def gaussfuction(sigma, ene_len=6, dt=1):
    """
    生成归一化高斯滤波器
    采样间隔1ns，6sigma原则包含99%能量

    参数：
    ----
    sigma: float
        高斯sigma
    ene_len: int
        能量长度系数（默认6）
    dt: int
        采样间隔（默认1）

    返回：
    ----
    dataoutput: np.ndarray
        归一化高斯滤波器
    """
    samplepoint = np.ceil(sigma * ene_len / 2).astype(int)
    i = np.arange(1, 2 * samplepoint + 2, dt)
    center = samplepoint + 1
    fildata = np.exp(-(i - center) ** 2 / (2 * sigma ** 2))
    sumfil = np.sum(fildata)
    dataoutput = fildata / sumfil  # 归一化
    return dataoutput


def fit_gauss12(para, signal, Inflection):
    """
    计算高斯拟合与原始信号的残差

    参数：
    ----
    para: np.ndarray
        高斯参数（展平或2D数组）
    signal: np.ndarray
        原始信号
    Inflection: list
        信号范围 [起始, 终止]

    返回：
    ----
    diff: np.ndarray
        残差数组
    """
    if para.ndim == 1:
        para = para.reshape(-1, 3)

    start_idx = int(Inflection[0])
    end_idx = int(Inflection[1])
    x = np.arange(1, len(signal) + 1)
    y = signal[start_idx:end_idx + 1]

    gauss = np.zeros(len(x))
    for i in range(len(para)):
        amp, pos, sigma = para[i]
        gauss += amp * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))

    # 只返回有效区间的残差
    diff = y - gauss[start_idx:end_idx + 1]
    return diff.flatten()


# 测试示例（可选）
if __name__ == "__main__":
    # 生成测试波形（2个高斯分量叠加）
    x = np.arange(1, 200)
    y1 = 10 * np.exp(-(x - 50) ** 2 / (2 * 8 ** 2))
    y2 = 8 * np.exp(-(x - 100) ** 2 / (2 * 10 ** 2))
    y = y1 + y2 + np.random.normal(0, 0.5, len(x))  # 加噪声

    # 调用波形分解函数
    prfnl, prini = waveresolve(x, y, filtwidth=4, noise_sigma=0.5, txsigma=4, maxwavenum=2, display=1)
    print("初始估计参数：")
    print(prini)
    print("\n最终拟合参数：")
    print(prfnl)