from abc import ABC, abstractmethod  # 导入ABC和abstractmethod，用于定义抽象基类和抽象方法
from typing import Any  # 从typing模块导入Any类型，用于类型注解

from vidur.types import BaseIntEnum  # 从vidur.types模块导入BaseIntEnum，用作键的枚举类型

# 定义一个抽象基类BaseRegistry
class BaseRegistry(ABC):
    _key_class = BaseIntEnum  # 定义一个类属性_key_class，默认设为BaseIntEnum
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # 调用父类的__init_subclass__方法
        cls._registry = {}  # 为子类创建一个类属性_registry，用于存储注册的实现类
    
    @classmethod
    def register(cls, key: BaseIntEnum, implementation_class: Any) -> None:
        if key in cls._registry:  # 如果key已经存在于注册表中，返回
            return

        cls._registry[key] = implementation_class  # 否则将实现类添加到注册表中
    
    @classmethod
    def unregister(cls, key: BaseIntEnum) -> None:
        if key not in cls._registry:  # 如果key不在注册表中，抛出异常
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]  # 否则从注册表中删除该key
    
    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if key not in cls._registry:  # 如果key不在注册表中，抛出异常
            raise ValueError(f"{key} is not registered")

        return cls._registry[key](*args, **kwargs)  # 从注册表中获取实现类的实例，传递参数*args和**kwargs
    
    @classmethod
    def get_class(cls, key: BaseIntEnum) -> Any:
        if key not in cls._registry:  # 如果key不在注册表中，抛出异常
            raise ValueError(f"{key} is not registered")

        return cls._registry[key]  # 从注册表中获取实现类
    
    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> BaseIntEnum:
        pass  # 抽象方法，子类需要实现此方法，根据字符串返回相应的BaseIntEnum键

    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)  # 根据key_str获取BaseIntEnum键并获取对应的实例