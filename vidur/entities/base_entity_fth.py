class BaseEntity:
    _id = -1  # 定义类变量_id，初始值为-1

    @classmethod
    def generate_id(cls):
        cls._id += 1  # 类方法generate_id，每次调用时将类变量_id加1
        return cls._id  # 返回_id的当前值

    @property
    def id(self) -> int:
        return self._id  # 定义实例的id属性，返回类变量_id的值

    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        # 使用to_dict方法获取对象的字典表示，并将其转换为字符串
        class_name = self.__class__.__name__  # 获取当前实例的类名
        return f"{class_name}({str(self.to_dict())})"  # 返回格式化的字符串，包含类名和对象的字典表示