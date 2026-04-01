from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from PhotonDenoise import PDSE
from ATL08Process import preprocess_atl08
def main(atl08_file_path, beam='gt1l', n_trees=500, mtry=4):
    """
    主流程：预处理→去噪→分类→精度评估
    :param atl08_file_path: ATL08文件路径
    :param beam: 地面轨道（gt1l/gt1r/gt2l/gt2r/gt3l/gt3r）
    :param n_trees: 随机森林树数量（论文设为500）
    :param mtry: 每棵树随机选择的特征数（论文设为4）
    """
    # 1. 数据预处理（读取ATL08+提取特征+标签）
    print("第一步：数据预处理...")
    df, features, cloud = preprocess_atl08(atl08_file_path, beam)
    print(f"预处理后有效样本数：{len(df)}")
    print(f"四类地表样本分布：{df['label'].value_counts().sort_index().tolist()}")

    # 2. 椭圆密度去噪（PDSE）
    print("第二步：椭圆密度去噪（PDSE）...")
    denoiser = PDSE()
    signal_cloud = denoiser.denoise(cloud)
    print(f"去噪后信号光子数：{len(signal_cloud)}")

    # 3. 数据集划分（论文：25%训练，75%测试，重复5次取平均）
    print("第三步：随机森林分类...")
    overall_accs = []
    kappa_scores = []
    reports = []

    for i in range(5):  # 重复5次（论文要求）
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df['label'], test_size=0.75, random_state=i, stratify=df['label']
        )

        # 初始化随机森林（匹配论文参数）
        rfc = RandomForestClassifier(
            n_estimators=n_trees,
            max_features=mtry,
            random_state=i,
            n_jobs=-1
        )
        rfc.fit(X_train, y_train)

        # 预测与评估
        y_pred = rfc.predict(X_test)
        overall_acc = overall_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['水体', '森林', '低植被', '城市/裸地'],
                                       output_dict=True)

        overall_accs.append(overall_acc)
        kappa_scores.append(kappa)
        reports.append(report)
        print(f"第{i + 1}次训练 - 总体精度：{overall_acc:.4f}，Kappa系数：{kappa:.4f}")

    # 4. 结果汇总（论文要求的5次平均）
    avg_acc = np.mean(overall_accs)
    avg_kappa = np.mean(kappa_scores)
    print("\n===== 5次重复实验平均结果 =====")
    print(f"平均总体精度：{avg_acc:.4f}（论文目标：≥85%）")
    print(f"平均Kappa系数：{avg_kappa:.4f}（论文目标：≥70%）")

    # 5. 三类地表分类（合并低植被+城市/裸地）
    print("\n第四步：三类地表分类（水体/森林/低植被+城市/裸地）...")
    df['label_3class'] = df['label'].apply(lambda x: 2 if x in [2, 3] else x)
    overall_accs_3class = []
    kappa_scores_3class = []

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df['label_3class'], test_size=0.75, random_state=i, stratify=df['label_3class']
        )
        rfc_3class = RandomForestClassifier(
            n_estimators=n_trees, max_features=mtry, random_state=i, n_jobs=-1
        )
        rfc_3class.fit(X_train, y_train)
        y_pred = rfc_3class.predict(X_test)
        overall_accs_3class.append(overall_accuracy_score(y_test, y_pred))
        kappa_scores_3class.append(cohen_kappa_score(y_test, y_pred))

    avg_acc_3class = np.mean(overall_accs_3class)
    avg_kappa_3class = np.mean(kappa_scores_3class)
    print(f"三类分类平均总体精度：{avg_acc_3class:.4f}（论文目标：≥90%）")
    print(f"三类分类平均Kappa系数：{avg_kappa_3class:.4f}（论文目标：≥79%）")

    # 6. 可视化（混淆矩阵+特征重要性）
    print("\n第五步：结果可视化...")
    # 混淆矩阵（最后一次实验结果）
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['水体', '森林', '低植被+城市/裸地'],
                yticklabels=['水体', '森林', '低植被+城市/裸地'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('三类地表分类混淆矩阵')
    plt.show()

    # 特征重要性（论文表1属性排序）
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rfc_3class.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('特征重要性排序（论文10类属性）')
    plt.show()

    return avg_acc, avg_kappa, avg_acc_3class, avg_kappa_3class


# 运行主流程
if __name__ == "__main__":
    # 替换为你的ATL08文件路径（HDF5格式）
    ATL08_FILE = r"D:\研究生\SanFrancisco\ICESAT\ATL08_20251110081832_08672906_007_01_subsetted.h5"
    # 选择地面轨道（如gt1l，可替换为其他轨道）
    BEAM = "gt1l"

    # 执行分类流程
    avg_acc_4class, avg_kappa_4class, avg_acc_3class, avg_kappa_3class = main(ATL08_FILE, BEAM)