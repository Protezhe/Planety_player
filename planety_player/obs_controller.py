from typing import Optional

import obsws_python as obs


class OBSController:
    """Control OBS via WebSocket."""

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self.client: Optional[obs.ReqClient] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to OBS WebSocket."""
        try:
            self.client = obs.ReqClient(host=self.host, port=self.port, password=self.password)
            self._connected = True
            print(f"✓ Connected to OBS WebSocket at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to OBS: {e}")
            print("  Make sure OBS is running and WebSocket server is enabled")
            print("  (Tools → WebSocket Server Settings)")
            self._connected = False
            return False

    def switch_scene(self, scene_name: str) -> bool:
        """Switch to a different scene."""
        if not self._connected or not self.client:
            print("✗ Cannot switch scene: not connected to OBS")
            return False
        try:
            self.client.set_current_program_scene(scene_name)
            print(f"✓ Switched to scene: {scene_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to switch scene to '{scene_name}': {e}")
            return False

    def get_current_scene(self) -> Optional[str]:
        """Get current scene name."""
        if not self._connected or not self.client:
            return None
        try:
            response = self.client.get_current_program_scene()
            return response.current_program_scene_name
        except Exception as e:
            print(f"✗ Failed to get current scene: {e}")
            return None

    def get_scene_list(self) -> list:
        """Get list of all available scenes."""
        if not self._connected or not self.client:
            return []
        try:
            response = self.client.get_scene_list()
            return [scene["sceneName"] for scene in response.scenes]
        except Exception as e:
            print(f"✗ Failed to get scene list: {e}")
            return []

    def set_preview_scene(self, scene_name: str) -> bool:
        """Set preview scene (Studio Mode only)."""
        if not self._connected or not self.client:
            print("✗ Cannot set preview scene: not connected to OBS")
            return False
        try:
            self.client.set_current_preview_scene(scene_name)
            print(f"✓ Set preview scene to: {scene_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to set preview scene to '{scene_name}': {e}")
            return False

    def get_studio_mode_enabled(self) -> bool:
        """Check if Studio Mode is enabled."""
        if not self._connected or not self.client:
            return False
        try:
            response = self.client.get_studio_mode_enabled()
            return response.studio_mode_enabled
        except Exception as e:
            print(f"✗ Failed to check studio mode: {e}")
            return False

    def disconnect(self):
        """Disconnect from OBS."""
        if self.client:
            self.client.disconnect()
            self._connected = False
            print("✓ Disconnected from OBS")
