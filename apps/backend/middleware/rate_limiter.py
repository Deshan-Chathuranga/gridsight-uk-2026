import time
from cachetools import TTLCache
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to rate limit requests on /api endpoints to prevent abuse.
    Uses an in-memory TTLCache with a sliding window to automatically clean up old entries.
    """
    def __init__(self, app):
        super().__init__(app)
        # Using a TTLCache keeps memory usage bounded (max 10000 client-endpoint combinations)
        # and auto-expires entries after 1 hour (3600 seconds), which matches our longest window.
        self.cache = TTLCache(maxsize=10000, ttl=3600)
        
    def get_limit_and_window(self, path: str):
        """Returns (limit, window_seconds, group_name) for a given path."""
        if path.startswith("/api/pipeline/sync"):
            return 5, 3600, "pipeline_sync"
        elif path.startswith("/api/pipeline"):
            return 30, 60, "pipeline_other"
        elif path.startswith("/api/forecasts") or path.startswith("/api/xai"):
            return 60, 60, "api_data"
        else:
            return 100, 60, "default"

    async def dispatch(self, request: Request, call_next):
        # Only rate limit API endpoints
        if not request.url.path.startswith("/api"):
            return await call_next(request)
            
        # Extract client IP, checking for X-Forwarded-For header first
        ip = request.headers.get("x-forwarded-for")
        if ip:
            ip = ip.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
            
        path = request.url.path
        
        # Route-specific rate limits
        limit, window, group = self.get_limit_and_window(path)
            
        key = (ip, group)
        now = time.time()
        
        # Get request history for the client + route group
        history = self.cache.get(key, [])
        # Only keep requests inside the time window
        history = [t for t in history if now - t < window]
        
        if len(history) >= limit:
            retry_after = int(window - (now - history[0]))
            if retry_after <= 0:
                retry_after = 1
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Too many requests. Please try again later.",
                    "retry_after_seconds": retry_after
                },
                headers={
                    "Retry-After": str(retry_after)
                }
            )
            
        history.append(now)
        self.cache[key] = history
        
        return await call_next(request)
