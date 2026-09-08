"""ch08 内置工具插件目录:落一个 .py 文件(暴露 register(ctx))即完成注册,不动核心代码。

启动时按文件名序 import 跳过 `_` 前缀;ctx 为 ToolContext(session_factory/service/kb/top_k)。
"""
import importlib
import logging
from pathlib import Path

from app.tools.base import ToolContext, ToolRegistryV2

logger = logging.getLogger(__name__)


def load_plugins(registry: ToolRegistryV2, ctx: ToolContext,
                 directory: Path | None = None) -> list[str]:
    """装载插件目录(缺省本包所在目录);directory 参数供测试注入临时目录。"""
    import importlib.util

    loaded = []
    for path in sorted((directory or Path(__file__).parent).glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.warning("插件 %s 加载失败", path.name, exc_info=True)
            continue
        hook = getattr(module, "register", None)
        if hook is None:
            logger.warning("插件 %s 缺 register(ctx) 钩子,跳过", path.name)
            continue
        try:
            hook(registry, ctx)
        except Exception:
            logger.warning("插件 %s 注册失败,跳过", path.name, exc_info=True)
            continue
        loaded.append(path.stem)
    if loaded:
        logger.info("已装载工具插件:%s", ", ".join(loaded))
    return loaded
