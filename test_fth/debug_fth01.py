
import pdb  # 导入Python标准库中的pdb模块，该模块用于调试Python代码

def add1(a, b):  # 定义一个名为add1的函数，它接收两个参数a和b
    pdb.set_trace()  # 调用pdb中的set_trace方法，程序将在这里暂停，进入调试模式
    c = int(a) + int(b)  # 将参数a和b转换为整数，并将它们的和赋值给变量c
    return "sum : {}".format(c)  # 返回一个格式化的字符串，表示这两个数的和

print(add1(3, 5))  # 调用add1函数，传入3和5作为参数，并打印该函数的返回值


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