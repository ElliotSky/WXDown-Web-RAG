from __future__ import annotations


class WxchatragError(Exception):
    """wxchatrag 顶层自定义异常基类。"""


class ConfigurationError(WxchatragError):
    """配置错误，例如路径不存在、环境变量缺失等。"""


class DataSourceNotFoundError(WxchatragError):
    """知识库数据源不存在或不可读。"""


class VectorStoreNotFoundError(WxchatragError):
    """向量库不存在，需要先执行 ingest。"""


class EmptyQuestionError(WxchatragError):
    """用户输入的问题为空。"""



