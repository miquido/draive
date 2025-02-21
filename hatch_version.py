import os
from subprocess import check_output

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomVersionHook(MetadataHookInterface):
    def update(self, metadata):
        if version := os.getenv("PROJECT_VERSION"):
            metadata["version"] = version

        else:
            try:
                metadata["version"] = check_output(
                    ["git", "describe", "--tags", "--abbrev=0"],
                    text=True,
                ).strip()

            except Exception:
                metadata["version"] = "0.0.0"
