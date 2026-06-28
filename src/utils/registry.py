"""
Registry util to map string names to class/function objects.

[设计模式] 注册表模式 (Registry Pattern)

背景：
在深度学习项目中，我们经常需要通过配置文件 (YAML/JSON) 来选择不同的模型、优化器或数据增强策略。
例如，在 YAML 中写 `backbone: type: "resnet50"`，代码就需要自动实例化 `ResNet50` 类。

问题：
如果写大量的 `if type == "resnet50": ... elif type == "vit": ...`，代码会变得非常臃肿且难以维护。
每新增一个模型，都需要修改工厂模式的 `if/else` 代码。

解决方案 - Registry：
1. 维护一个全局字典 (Map)，键是字符串名 (如 "resnet50")，值是对应的类 (Class)。
2. 使用装饰器 (Decorator) 在类定义时自动将其注册到字典中。
3. 通过字符串名查表即可获得类，实现完全解耦。

优化（2026-02-02）：
- ✅ 添加类型提示（Type Hints）提升 IDE 支持
- ✅ 支持泛型（Generic）提供类型安全
- ✅ 增强错误提示信息

使用示例：
    # 1. 定义注册表
    MODELS = Registry("models")
    
    # 2. 注册类
    @MODELS.register("my_model")
    class MyModel: ...
    
    # 3. 通过字符串获取类并实例化
    model_cls = MODELS.get("my_model")
    model = model_cls(...)
"""

from typing import TypeVar, Generic, Type, Dict, Optional, Callable, Any

# 定义泛型类型变量
T = TypeVar('T')

class Registry(Generic[T]):
    """
    通用注册表类（支持泛型）。
    用于管理字符串到类/函数的映射。
    
    类型参数:
        T: 注册对象的基类型（用于类型提示）
        
    示例:
        # 定义类型安全的注册表
        BACKBONES: Registry[nn.Module] = Registry("backbone")
        
        # 注册时自动类型检查
        @BACKBONES.register("resnet50")
        class ResNet50(nn.Module):
            ...
    """
    def __init__(self, name: str) -> None:
        """
        参数:
            name: 注册表名称 (例如 "backbones", "losses")，主要用于报错信息。
        """
        self._name: str = name
        self._module_dict: Dict[str, Type[T]] = {}  # 核心存储结构：{"name": ClassObject}

    def __repr__(self) -> str:
        """返回注册表的字符串表示"""
        registered_items = list(self._module_dict.keys())
        return (f"{self.__class__.__name__}(name='{self._name}', "
                f"count={len(registered_items)}, items={registered_items})")

    @property
    def name(self) -> str:
        """返回注册表名称"""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type[T]]:
        """返回所有已注册的模块字典"""
        return self._module_dict
    
    def list_registered(self) -> list[str]:
        """
        列出所有已注册的名称
        
        返回:
            list[str]: 已注册名称列表
        """
        return list(self._module_dict.keys())

    def get(self, key: str) -> Optional[Type[T]]:
        """
        通过名称获取注册的对象。
        
        参数:
            key: 注册时的名称。
            
        返回:
            对应的类或函数对象。如果未找到则返回 None。
            
        抛出:
            KeyError: 如果 key 不存在且没有类似名称，提供建议
        """
        if key in self._module_dict:
            return self._module_dict[key]
        
        # 未找到时提供有用的错误信息
        available_keys = self.list_registered()
        
        # 尝试找到相似名称（简单字符串匹配）
        suggestions = [k for k in available_keys if key.lower() in k.lower() or k.lower() in key.lower()]
        
        error_msg = f"'{key}' is not registered in {self._name}. "
        if suggestions:
            error_msg += f"Did you mean one of: {suggestions}? "
        error_msg += f"Available: {available_keys}"
        
        raise KeyError(error_msg)

    def register(self, module_name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        """
        [装饰器] 用于注册模块。
        
        参数:
            module_name: 自定义注册名称（可选）
        
        返回:
            装饰器函数
        
        用法:
            @registry.register()                  # 使用类名作为注册名
            class MyModel: ...
            
            @registry.register("custom_name")    # 指定自定义注册名
            class MyModel: ...
        """
        def _register(module: Type[T]) -> Type[T]:
            # 确定注册名：如果用户未指定，则默认使用类名 (module.__name__)
            name = module_name if module_name is not None else module.__name__
            
            # 检查是否重复注册，防止覆盖
            if name in self._module_dict:
                existing_module = self._module_dict[name]
                raise KeyError(
                    f"'{name}' is already registered in {self._name}. "
                    f"Existing: {existing_module.__module__}.{existing_module.__name__}, "
                    f"New: {module.__module__}.{module.__name__}"
                )
            
            # 存入字典
            self._module_dict[name] = module
            return module

        return _register
    
    def build(self, cfg: Dict[str, Any]) -> T:
        """
        根据配置字典构建对象（便捷方法）
        
        参数:
            cfg: 配置字典，必须包含 'type' 键
            
        返回:
            实例化的对象
            
        示例:
            backbone_cfg = {"type": "resnet50", "pretrained": True}
            backbone = BACKBONES.build(backbone_cfg)
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"Config must be a dict, got {type(cfg)}")
        
        if 'type' not in cfg:
            raise KeyError(f"Config must contain 'type' key, got keys: {list(cfg.keys())}")
        
        obj_type = cfg.pop('type')
        obj_cls = self.get(obj_type)
        
        try:
            return obj_cls(**cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build {obj_type} from {self._name}: {e}"
            ) from e

# --- 全局注册表实例 ---
# 这些全局变量充当了不同组件的“目录”。
# 在 src/models/__init__.py 等文件中，我们会导入这些 Registry，
# 从而让分散在各个文件中的类能够注册进来。

BACKBONES = Registry("backbone")   # 骨干网络 (如 ResNet, Swin)
HEADS = Registry("head")           # 分类头 (如 Linear, MLP)
LOSSES = Registry("loss")          # 损失函数 (如 CrossEntropy)
DATASETS = Registry("dataset")     # 数据集 (如 ImageFolder)
TRANSFORMS = Registry("transform") # 数据增强 (如 RandomCrop)
MODELS = Registry("model")         # 整体模型 (如 ImageClassifier)
