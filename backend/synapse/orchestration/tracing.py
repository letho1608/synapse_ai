from dataclasses import dataclass
from typing import Dict, Optional, Any
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from contextlib import contextmanager
import time
from threading import Lock

@dataclass
class TraceContext:
  request_id: str
  sequence_number: int
  current_span: Optional[trace.Span] = None
  trace_parent: Optional[str] = None
  token_group_span: Optional[trace.Span] = None
  token_count: int = 0
  token_group_size: int = 10  # Default group size
  request_span: Optional[trace.Span] = None  # Track the main request span

class Tracer:
  def __init__(self):
    self.tracer = trace.get_tracer("synapse")
    self.contexts: Dict[str, TraceContext] = {}
    self._lock = Lock()
    self.propagator = TraceContextTextMapPropagator()
    


  def extract_context(self, trace_parent: str) -> Optional[context.Context]:
    """Extract span context from carrier"""
    if not trace_parent:
      return None
    carrier = {"traceparent": trace_parent}
    return self.propagator.extract(carrier)


  @contextmanager
  def start_span(self, name: str, context: TraceContext, extra_attributes: Optional[Dict[str, Any]] = None):
    """Start a new span with proper parent context"""
    attributes = {
      "request_id": context.request_id,
      "sequence_number": context.sequence_number
    }
    if extra_attributes:
      attributes.update(extra_attributes)
      
    # Use request span as parent if available
    parent_ctx = None
    if context.request_span:
      parent_ctx = trace.set_span_in_context(context.request_span)
    elif context.trace_parent:
      parent_ctx = self.extract_context(context.trace_parent)
      if parent_ctx and not context.request_span:
        # Create a new request span that links to the parent context
        context.request_span = self.tracer.start_span(
          "request",
          context=parent_ctx,
          attributes={
            "request_id": context.request_id,
            "sequence_number": context.sequence_number
          }
        )
        parent_ctx = trace.set_span_in_context(context.request_span)
    elif context.current_span:
      parent_ctx = trace.set_span_in_context(context.current_span)
    
    # Create span with parent context if it exists
    if parent_ctx:
      span = self.tracer.start_span(
        name,
        context=parent_ctx,
        attributes=attributes
      )
    else:
      span = self.tracer.start_span(
        name,
        attributes=attributes
      )
    
    # Update context with current span
    prev_span = context.current_span
    context.current_span = span
    
    try:
      start_time = time.perf_counter()
      yield span
      duration = time.perf_counter() - start_time
      span.set_attribute("duration_s", duration)
      span.set_status(Status(StatusCode.OK))
    except Exception as e:
      span.set_status(Status(StatusCode.ERROR, str(e)))
      raise
    finally:
      span.end()
      context.current_span = prev_span

# Global tracer instance
tracer = Tracer() 