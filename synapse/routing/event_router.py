from typing import Generic, TypeVar, Callable, Dict, List, Any
import asyncio
from dataclasses import dataclass
from loguru import logger

T = TypeVar("T")

@dataclass
class Event(Generic[T]):
    topic: str
    data: T
    origin: str
    timestamp: float

class EventRouter:
    """
    EventRouter manages P2P message distribution using a Pub/Sub model.
    Inspired by Exo's TopicRouter but optimized for Synapse's asyncio architecture.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Event], Any]]] = {}
        self._running = False

    def subscribe(self, topic: str, callback: Callable[[Event], Any]):
        """Register a callback for a specific topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Event], Any]):
        """Unregister a callback from a specific topic."""
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(callback)
                logger.debug(f"Unsubscribed from topic: {topic}")
            except ValueError:
                pass

    async def publish(self, topic: str, data: Any, origin: str = "local"):
        """
        Publish an event to local subscribers. 
        Supports wildcard matching for topics ending with /# (e.g., synapse/#).
        """
        event = Event(topic=topic, data=data, origin=origin, timestamp=asyncio.get_event_loop().time())
        
        # 1. Exact match
        callbacks = list(self.subscribers.get(topic, []))
        
        # 2. Wildcard matches (simple /#)
        for sub_topic, sub_callbacks in self.subscribers.items():
            if sub_topic.endswith("/#"):
                prefix = sub_topic[:-2]
                if topic.startswith(prefix):
                    callbacks.extend(sub_callbacks)
            elif sub_topic == "#": # Global wildcard
                callbacks.extend(sub_callbacks)

        if callbacks:
            tasks = []
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in sync callback for {topic}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self):
        """Initialize the router and prepare P2P mesh connection."""
        self._running = True

    async def stop(self):
        """Shutdown the router and clean up P2P connections."""
        self._running = False
        logger.info("Event Router stopped.")
