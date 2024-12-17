# import IPython
# print(IPython.__version__)

from IPython.display import display
import pandas as pd
 
# 显示一个字符串
display('Hello, world!')
 
# 显示一个 Pandas 数据框
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
display(df)
 
# 显示一张图片
from PIL import Image
img = Image.open('image.jpg')
display(img)
 
# 显示一段 HTML 代码
display('<h1>This is a heading</h1>')
