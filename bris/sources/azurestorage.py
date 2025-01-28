# python standard library
import logging
import os
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from typing import Optional

# PyPi packages
from dotenv import load_dotenv
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
except Exception as e:
    LOGGER.error(f"Could not import packages. Exception raised: {e}")

# Local packages

load_dotenv()


class AzureStorage:
    BLOB_NAME = None
    BLOB_CONTAINER_NAME = None
    PATH = None

    def __init__(
        self, *, args: Optional[Namespace] = None, config: Optional[DictConfig] = None
    ) -> None:

        self.args = args
        self.config = config

        self.BLOB_NAME = args.blob_name
        self.BLOB_CONTAINER_NAME = args.blob_container_name
        self.lagged = 1  # args.lagged if args.lagged else 1
        CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONTAINER_STRING")

        if not CONNECTION_STRING:
            LOGGER.error(
                "Could not fetch or find azure storage container string. Please check if it is with environment variable list"
            )
            raise ValueError(
                "AZURE_STORAGE_CONTAINER_STRING could not be found in the environment variable list"
            )
        self.blob_service_client = BlobServiceClient.from_connection_string(
            CONNECTION_STRING
        )
        self._check_dir

    @property
    def _check_dir(self):
        from pathlib import Path

        if not self.PATH:
            PATH = Path.cwd()
        else:
            PATH = Path(self.PATH)

        PATH = PATH / "input"
        if not PATH.exists():
            LOGGER.info(f"Input folder does path exists. Creating folder at: {PATH}")
            PATH.mkdir()

        self.PATH = PATH
        LOGGER.info("Input folder path exists, will not create folder")

    @cached_property
    def fetch_blobs(self) -> None:
        """
        Downloads data from azure blob storage
        """
        # TODO: make this compatible with azure functions
        # and azure key vault and not use .env for keys

        # fetch blob, container name and conn str
        BLOB_CONTAINER_NAME = self.BLOB_CONTAINER_NAME

        try:
            # TODO : implement so it fetches latest two initial zarr states
            container_client = self.blob_service_client.get_container_client(
                BLOB_CONTAINER_NAME
            )
            # blob_client = container_client.get_blob_client(self.blob_name)
            blobs = [
                {
                    "name": blob.name,
                    "container": blob.container,
                    "last_modified": blob.last_modified,
                }
                for blob in container_client.list_blobs()
            ]
            blobs.sort(key=lambda x: x["last_modified"], reverse=True)

            LOGGER.info("Successfully fetched blobs from azure blob storage")
            return blobs  # [:self.lagged]

        except Exception as e:
            LOGGER.error(f"Error fetching blob information. An exception occured!", exc_info=True)
            raise RuntimeError("Error fetching blob information.") from e

    @property
    def download_blob(self) -> None:
        blobs = self.fetch_blobs

        # for b in blobs:
        def _download_blob(b, blob_service_client):
            blob_client = blob_service_client.get_blob_client(
                container=b["container"], blob=b["name"]
            )
            try:
                file_path = self.PATH / b["name"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file=file_path, mode="wb") as currBlob:
                    download_stream = blob_client.download_blob()
                    currBlob.write(download_stream.readall())
            except Exception as e:
                LOGGER.error(f"Error in downloading blob to local filesystem.")
                raise RuntimeError("Error in downloading blob.") from e

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_download_blob, b, self.blob_service_client)
                for b in blobs
            ]
            for future in as_completed(futures):
                future.result()
        return
