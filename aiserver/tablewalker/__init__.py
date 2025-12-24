# -*- coding: utf-8 -*-
"""
TableWalker 后端模块

提供 DataFrame 数据转换和 HTML 生成功能，支持在 Jupyter 中直接调用 show_df() 函数显示 TableWalker 组件。
"""

from .api import show_df

__all__ = ['show_df']
