# utils/callbacks.py
import logging
from typing import Callable, Any, Dict

logger = logging.getLogger("noesis.callbacks")

class ServiceRegistry:
    """A runtime registry to execute interface actions without direct imports."""
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        cls._registry[name] = func
        logger.info(f"Registered service callback: '{name}'")

    @classmethod
    async def call(cls, name: str, *args, **kwargs) -> Any:
        if name not in cls._registry:
            logger.warning(f"Service callback '{name}' is not registered. Skipping execution.")
            return f"Error: Callback service '{name}' is currently unavailable."
        
        try:
            func = cls._registry[name]
            import inspect
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback service '{name}': {e}")
            raise
