"""
Asset management for STP providers.

Supports:
- Filesystem assets for stdio transport (via ASSET_PATH env var)
- HTTP assets for websocket transport (via asset_endpoint)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import os
import re
import uuid
import logging
import aiohttp

# Asset IDs are opaque tokens from a fixed charset; reject anything else before touching the disk.
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}\.[A-Za-z0-9]{1,12}$")


def valid_asset_id(asset_id: str) -> bool:
    return isinstance(asset_id, str) and bool(_ASSET_ID_RE.match(asset_id))

logger = logging.getLogger(__name__)


class AssetManager(ABC):
    """Abstract base class for asset management."""

    @abstractmethod
    async def upload(self, data: bytes, extension: str = ".bin") -> str:
        """
        Upload asset data and return an asset ID.

        Args:
            data: Binary data to upload
            extension: File extension (e.g., ".png", ".mp4")

        Returns:
            Asset ID that can be used to reference this asset
        """
        pass

    @abstractmethod
    async def download(self, asset_id: str) -> bytes:
        """
        Download asset data by ID.

        Args:
            asset_id: The asset ID to download

        Returns:
            Binary data of the asset
        """
        pass

    @abstractmethod
    async def delete(self, asset_id: str) -> bool:
        """
        Delete an asset by ID.

        Args:
            asset_id: The asset ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, asset_id: str) -> bool:
        """Check if an asset exists."""
        pass


class FilesystemAssetManager(AssetManager):
    """
    Asset manager using shared filesystem.

    Used with stdio transport. Assets are stored in ASSET_PATH directory.
    Both Stimma and the provider read/write directly to this directory.
    """

    def __init__(self, asset_path: Optional[str] = None):
        """
        Initialize filesystem asset manager.

        Args:
            asset_path: Path to asset directory. If None, uses ASSET_PATH env var.
        """
        self._asset_path = Path(asset_path or os.environ.get("ASSET_PATH", "/tmp/stimma-assets"))
        self._asset_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Filesystem asset manager using: {self._asset_path}")

    def _get_path(self, asset_id: str) -> Path:
        """Get the filesystem path for an asset ID."""
        # Sanitize asset_id to prevent path traversal
        safe_id = Path(asset_id).name
        return self._asset_path / safe_id

    async def upload(self, data: bytes, extension: str = ".bin") -> str:
        """Upload asset data to filesystem."""
        asset_id = f"{uuid.uuid4()}{extension}"
        path = self._get_path(asset_id)
        path.write_bytes(data)
        logger.debug(f"Uploaded asset: {asset_id} ({len(data)} bytes)")
        return asset_id

    async def download(self, asset_id: str) -> bytes:
        """Download asset data from filesystem."""
        # Check if it's an absolute path that exists (stdio mode passes full paths)
        if os.path.isabs(asset_id):
            abs_path = Path(asset_id)
            if abs_path.exists():
                data = abs_path.read_bytes()
                logger.debug(f"Downloaded from absolute path: {asset_id} ({len(data)} bytes)")
                return data

        # Otherwise look in the asset directory
        path = self._get_path(asset_id)
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_id}")
        data = path.read_bytes()
        logger.debug(f"Downloaded asset: {asset_id} ({len(data)} bytes)")
        return data

    async def delete(self, asset_id: str) -> bool:
        """Delete asset from filesystem."""
        path = self._get_path(asset_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted asset: {asset_id}")
            return True
        return False

    async def exists(self, asset_id: str) -> bool:
        """Check if asset exists on filesystem."""
        # Check absolute path first
        if os.path.isabs(asset_id):
            if Path(asset_id).exists():
                return True
        return self._get_path(asset_id).exists()

    @property
    def asset_path(self) -> Path:
        """Get the asset directory path."""
        return self._asset_path


class HttpAssetManager(AssetManager):
    """
    Asset manager using HTTP endpoints.

    Used with websocket transport. Assets are uploaded/downloaded via HTTP
    to the asset_endpoint provided during registration.
    """

    def __init__(
        self,
        base_url: str,
        asset_endpoint: str,
        auth_token: Optional[str] = None,
    ):
        """
        Initialize HTTP asset manager.

        Args:
            base_url: Base URL of the Stimma server (e.g., "https://stimma.example.com")
            asset_endpoint: Asset endpoint path (e.g., "/tools-rpc/assets")
            auth_token: Optional bearer token for authentication
        """
        self._base_url = base_url.rstrip("/")
        self._asset_endpoint = asset_endpoint.rstrip("/")
        self._auth_token = auth_token
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _get_url(self, asset_id: str) -> str:
        """Get the full URL for an asset ID."""
        return f"{self._base_url}{self._asset_endpoint}/{asset_id}"

    async def upload(self, data: bytes, extension: str = ".bin") -> str:
        """Upload asset data via HTTP PUT."""
        asset_id = f"{uuid.uuid4()}{extension}"
        url = self._get_url(asset_id)

        # Determine content type from extension
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".bin": "application/octet-stream",
        }
        content_type = content_types.get(extension.lower(), "application/octet-stream")

        session = await self._get_session()
        async with session.put(url, data=data, headers={"Content-Type": content_type}) as resp:
            if resp.status not in (200, 201, 204):
                text = await resp.text()
                raise RuntimeError(f"Failed to upload asset: {resp.status} {text}")

        logger.debug(f"Uploaded asset: {asset_id} ({len(data)} bytes)")
        return asset_id

    async def download(self, asset_id: str) -> bytes:
        """Download asset data via HTTP GET."""
        url = self._get_url(asset_id)

        session = await self._get_session()
        async with session.get(url) as resp:
            if resp.status == 404:
                raise FileNotFoundError(f"Asset not found: {asset_id}")
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to download asset: {resp.status} {text}")
            data = await resp.read()

        logger.debug(f"Downloaded asset: {asset_id} ({len(data)} bytes)")
        return data

    async def delete(self, asset_id: str) -> bool:
        """Delete asset via HTTP DELETE."""
        url = self._get_url(asset_id)

        session = await self._get_session()
        async with session.delete(url) as resp:
            if resp.status == 404:
                return False
            if resp.status not in (200, 204):
                text = await resp.text()
                raise RuntimeError(f"Failed to delete asset: {resp.status} {text}")

        logger.debug(f"Deleted asset: {asset_id}")
        return True

    async def exists(self, asset_id: str) -> bool:
        """Check if asset exists via HTTP HEAD."""
        url = self._get_url(asset_id)

        session = await self._get_session()
        async with session.head(url) as resp:
            return resp.status == 200

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class LocalAssetServer(AssetManager):
    """
    Asset manager that stores assets locally and serves them via HTTP.

    Used with WebSocket transport. The provider hosts assets and Stimma
    uploads/downloads them via HTTP.
    """

    def __init__(self, asset_path: Optional[str] = None):
        """
        Initialize local asset server.

        Args:
            asset_path: Path to asset directory. If None, uses a temp directory.
        """
        import tempfile
        if asset_path:
            self._asset_path = Path(asset_path)
        else:
            self._asset_path = Path(tempfile.mkdtemp(prefix="stimma-provider-assets-"))
        self._asset_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local asset server using: {self._asset_path}")

    def _get_path(self, asset_id: str) -> Path:
        """Get the filesystem path for an asset ID."""
        # Sanitize asset_id to prevent path traversal
        safe_id = Path(asset_id).name
        return self._asset_path / safe_id

    async def upload(self, data: bytes, extension: str = ".bin") -> str:
        """Upload asset data to local storage."""
        asset_id = f"{uuid.uuid4()}{extension}"
        path = self._get_path(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug(f"Uploaded asset: {asset_id} ({len(data)} bytes)")
        return asset_id

    async def download(self, asset_id: str) -> bytes:
        """Download asset data from local storage."""
        path = self._get_path(asset_id)
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_id}")
        data = path.read_bytes()
        logger.debug(f"Downloaded asset: {asset_id} ({len(data)} bytes)")
        return data

    async def delete(self, asset_id: str) -> bool:
        """Delete asset from local storage."""
        path = self._get_path(asset_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted asset: {asset_id}")
            return True
        return False

    async def exists(self, asset_id: str) -> bool:
        """Check if asset exists in local storage."""
        return self._get_path(asset_id).exists()

    @property
    def asset_path(self) -> Path:
        """Get the asset directory path."""
        return self._asset_path

    def get_aiohttp_routes(self, endpoint_path: str = "/assets"):
        """
        Get aiohttp routes for serving assets.

        Args:
            endpoint_path: The path prefix for asset routes (e.g., "/assets")

        Returns:
            List of aiohttp route definitions to add to an app

        CORS: handled by an upstream middleware in the plugin's startup
        (see stp_server/startup.py:_make_stp_cors_bypass_middleware).
        The middleware attaches Access-Control-* headers to responses
        and answers OPTIONS preflight, so the handlers below don't
        need to know anything about CORS.
        """
        from aiohttp import web

        def _reject_invalid(request: "web.Request"):
            if not valid_asset_id(request.match_info["asset_id"]):
                return web.Response(status=400, text="Malformed asset_id")
            return None

        async def handle_get(request: web.Request) -> web.Response:
            """Handle GET /assets/{asset_id}"""
            if (bad := _reject_invalid(request)) is not None:
                return bad
            asset_id = request.match_info["asset_id"]
            try:
                data = await self.download(asset_id)
                # Determine content type from extension
                ext = Path(asset_id).suffix.lower()
                content_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".mp4": "video/mp4",
                    ".webm": "video/webm",
                }
                content_type = content_types.get(ext, "application/octet-stream")
                return web.Response(body=data, content_type=content_type)
            except FileNotFoundError:
                return web.Response(status=404, text=f"Asset not found: {asset_id}")

        async def handle_put(request: web.Request) -> web.Response:
            """Handle PUT /assets/{asset_id} — streams to disk for large files."""
            if (bad := _reject_invalid(request)) is not None:
                return bad
            asset_id = request.match_info["asset_id"]
            path = self._get_path(asset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with open(path, "wb") as f:
                async for chunk in request.content.iter_chunked(64 * 1024):
                    f.write(chunk)
                    total += len(chunk)
            logger.debug(f"Received asset via HTTP PUT: {asset_id} ({total} bytes)")
            return web.Response(status=201, text="Created")

        async def handle_delete(request: web.Request) -> web.Response:
            """Handle DELETE /assets/{asset_id}"""
            if (bad := _reject_invalid(request)) is not None:
                return bad
            asset_id = request.match_info["asset_id"]
            if await self.delete(asset_id):
                return web.Response(status=204)
            return web.Response(status=404, text=f"Asset not found: {asset_id}")

        async def handle_head(request: web.Request) -> web.Response:
            """Handle HEAD /assets/{asset_id}"""
            if (bad := _reject_invalid(request)) is not None:
                return bad
            asset_id = request.match_info["asset_id"]
            if await self.exists(asset_id):
                return web.Response(status=200)
            return web.Response(status=404)

        # Return route definitions
        return [
            web.get(f"{endpoint_path}/{{asset_id}}", handle_get),
            web.put(f"{endpoint_path}/{{asset_id}}", handle_put),
            web.delete(f"{endpoint_path}/{{asset_id}}", handle_delete),
            web.head(f"{endpoint_path}/{{asset_id}}", handle_head),
        ]

    def cleanup(self) -> None:
        """Remove all assets and the asset directory."""
        import shutil
        if self._asset_path.exists():
            shutil.rmtree(self._asset_path)
            logger.info(f"Cleaned up asset directory: {self._asset_path}")


def create_asset_manager(
    transport_type: str,
    asset_path: Optional[str] = None,
    base_url: Optional[str] = None,
    asset_endpoint: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> AssetManager:
    """
    Create the appropriate asset manager based on transport type.

    Args:
        transport_type: "stdio" or "websocket"
        asset_path: For stdio, path to asset directory
        base_url: For websocket, base URL of Stimma server
        asset_endpoint: For websocket, asset endpoint path
        auth_token: For websocket, optional bearer token

    Returns:
        Configured AssetManager instance
    """
    if transport_type == "stdio":
        return FilesystemAssetManager(asset_path)
    elif transport_type == "websocket":
        # For WebSocket, use local asset server (provider hosts assets)
        return LocalAssetServer(asset_path)
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
