import simplekml
#定义矩形的四个角点，顺序需要闭合
lat=[37.5980333,37.8000083,37.8201833,37.6015667,37.5980333]
lon=[-122.5294333,-122.51785,-122.3875944,-122.3566333,-122.5294333]
#创建一个KML对象
kml=simplekml.Kml()
#创建一个多边形
poly=kml.newpolygon(name='Rectangle')

#simplekml是先经度后纬度
coordinates=list(zip(lon,lat))
poly.outerboundaryis=coordinates
#设计样式：黄色线条，宽度为2
poly.style.linestyle.width=2
poly.style.linestyle.color=simplekml.Color.yellow
poly.style.polystyle.fill=0

#保存kml文件
kml.save("C:\\Users\\Administrator\\Desktop\\flo.kml")