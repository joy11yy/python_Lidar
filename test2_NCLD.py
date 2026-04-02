import rasterio
from rasterio.transform import Affine

tif_path = r"D:\研究生\SanFrancisco\NCLD\Annual_NLCD_LndCov_2024_CU_C1V1_de325bd4-e7ec-4ccb-81ba-fddd2e8a3af3.tiff"

print("正在尝试读取地理信息...")
try:
    with rasterio.open(tif_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")

        # 检查是否是单位矩阵
        if src.transform == Affine(1, 0, 0, 0, 1, 0):
            print("\n❌ 警告：虽然打开了文件，但 Transform 仍然是单位矩阵！")
            print("   这意味着 .aux.xml 没有被成功加载。")
            print("   尝试解决方法：见下方的 '方案 2'。")
        else:
            print("\n✅ 成功！地理信息已加载。")
            print(f"   左上角坐标：{src.transform * (0, 0)}")
except Exception as e:
    print(f"❌ 打开出错: {e}")