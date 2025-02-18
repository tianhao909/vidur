from enum import IntEnum

# 定义形状枚举: ShapeEnum 用于给形状分配一个整数值，这便于在注册表中使用。
# 实现形状类: Circle、Square 和 Triangle 类分别代表不同形状，每个类都有计算面积的方法。
# 注册表类: ShapeFactory 类包含一个 _registry 属性，维护形状类型到实现类之间的映射。
# 使用 get 方法:
# 调用 ShapeFactory.get 获取实例时，传入 ShapeEnum 的值以及形状所需的参数。
# 如果 key 不存在，抛出 ValueError。
# 若 key 存在，则从 _registry 中获取对应的类，并使用传入的参数创建一个实例。
# 错误处理: 通过 try-except 块处理可能的 ValueError。
# 这种结构令代码具有高度的扩展性，便于添加新形状而不必修改现有逻辑。只需更新 _registry 字典即可。

# 形状的枚举类  定义形状枚举: ShapeEnum 用于给形状分配一个整数值，这便于在注册表中使用。
class ShapeEnum(IntEnum):
    CIRCLE = 1  # 定义枚举值CIRCLE，它的值是1
    SQUARE = 2  # 定义枚举值SQUARE，它的值是2
    TRIANGLE = 3  # 定义枚举值TRIANGLE，它的值是3

# 各种形状类 实现形状类: Circle、Square 和 Triangle 类分别代表不同形状，每个类都有计算面积的方法。
class Circle:
    def __init__(self, radius):  # 初始化方法，接受一个radius参数
        self.radius = radius  # 设置实例的半径属性
    
    def area(self):  # 计算圆的面积
        return 3.1415 * self.radius * self.radius  # 使用圆的面积公式：πr²

class Square:
    def __init__(self, side):  # 初始化方法，接受一个side参数
        self.side = side  # 设置实例的边长属性
    
    def area(self):  # 计算正方形的面积
        return self.side * self.side  # 使用正方形的面积公式：边长²

class Triangle:
    def __init__(self, base, height):  # 初始化方法，接受base和height参数
        self.base = base  # 设置实例的底边属性
        self.height = height  # 设置实例的高度属性
    
    def area(self):  # 计算三角形的面积
        return 0.5 * self.base * self.height  # 使用三角形的面积公式：0.5 * 底 * 高

# 注册表类 假设我们有一个处理不同形状的几何图形的注册表类： 注册表类: ShapeFactory 类包含一个 _registry 属性，维护形状类型到实现类之间的映射。
class ShapeFactory:
    _registry = {  # 定义一个类属性_registry，用于存储形状类的注册表
        ShapeEnum.CIRCLE: Circle,  # 将枚举值CIRCLE与Circle类关联
        ShapeEnum.SQUARE: Square,  # 将枚举值SQUARE与Square类关联
        ShapeEnum.TRIANGLE: Triangle,  # 将枚举值TRIANGLE与Triangle类关联
    }
    # 这个 get 类方法是从注册表中获取与给定 key 相关联的实现类实例。如果 key 不在注册表中，它会抛出一个 ValueError 异常。方法使用了 *args 和 **kwargs 来传递参数，以便灵活地生成类实例。
    # 要理解这个函数如何使用，我们需要假设一个包含 _registry 类属性的类，这个属性是一个字典，用于存储 key 和类的关联。
    @classmethod
    def get(cls, key: ShapeEnum, *args, **kwargs):  # 定义一个类方法，获取对应的形状类实例
        if key not in cls._registry:  # 检查键是否在注册表中
            raise ValueError(f"{key} is not registered")  # 如果不在，抛出异常 如果 key 不存在，抛出 ValueError。
        return cls._registry[key](*args, **kwargs)  # 使用提供的参数创建并返回相应的形状实例 若 key 存在，则从 _registry 中获取对应的类，并使用传入的参数创建一个实例。

# 使用示例
try:
    circle = ShapeFactory.get(ShapeEnum.CIRCLE, radius=5)  # 创建一个Circle实例，半径为5
    print("Circle area:", circle.area())  # 输出圆的面积
    
    square = ShapeFactory.get(ShapeEnum.SQUARE, side=4)  # 创建一个Square实例，边长为4
    print("Square area:", square.area())  # 输出正方形的面积
    
    triangle = ShapeFactory.get(ShapeEnum.TRIANGLE, base=3, height=4)  # 创建一个Triangle实例，底为3，高为4
    print("Triangle area:", triangle.area())  # 输出三角形的面积
    
except ValueError as e:  # 捕获并处理可能的ValueError异常 
    print(e)  # 输出异常信息

# 这种结构令代码具有高度的扩展性，便于添加新形状而不必修改现有逻辑。只需更新 _registry 字典即可。


# import pdb  # 导入Python标准库中的pdb模块，该模块用于调试Python代码

# def add1(a, b):  # 定义一个名为add1的函数，它接收两个参数a和b
#     pdb.set_trace()  # 调用pdb中的set_trace方法，程序将在这里暂停，进入调试模式
#     c = int(a) + int(b)  # 将参数a和b转换为整数，并将它们的和赋值给变量c
#     return "sum : {}".format(c)  # 返回一个格式化的字符串，表示这两个数的和

# print(add1(3, 5))  # 调用add1函数，传入3和5作为参数，并打印该函数的返回值


# import debugpy



# # # Listen for debugger connections on a specified port
# # debugpy.listen(5678)
# # # Wait for a debugger client to attach
# # debugpy.wait_for_client()

# def calculate_sum(a, b):
#     result = a + b  # Set a breakpoint here
#     return result

# # Call the function and print the result
# print(calculate_sum(10, 20))