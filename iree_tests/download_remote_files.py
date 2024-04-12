# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.storage.blob import BlobClient
from pathlib import Path
import argparse
import hashlib
import mmap
import pyjson5
import re

THIS_DIR = Path(__file__).parent


def check_azure_remote_file_matching_local(
    blob_client: BlobClient, local_file_path: Path
):
    if not local_file_path.exists():
        return False

    # Ask Azure for the md5 hash of the remote file.
    properties = blob_client.get_blob_properties()
    content_settings = properties.get("content_settings")
    if not content_settings:
        return False
    remote_md5 = content_settings.get("content_md5")

    # Compute the md5 hash of the local file.
    with open(local_file_path) as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as file:
        local_md5 = hashlib.md5(file).digest()
        return local_md5 == remote_md5


def download_azure_remote_file(test_dir: Path, remote_file: str):
    remote_file_name = remote_file.rsplit("/", 1)[-1]
    relative_dir = test_dir.relative_to(THIS_DIR)

    # Extract path components from Azure URL to use with the Azure Storage Blobs
    # client library for Python (https://pypi.org/project/azure-storage-blob/).
    #
    # For example:
    #   https://sharkpublic.blob.core.windows.net/sharkpublic/path/to/blob.txt
    #                                            ^           ^
    #   account_url:    https://sharkpublic.blob.core.windows.net
    #   container_name: sharkpublic
    #   blob_name:      path/to/blob.txt
    #
    # Note: we could also use the generic handler (e.g. wget, 'requests'), but
    # the client library offers other APIs.

    result = re.search(r"(https.+\.net)/([^/]+)/(.+)", remote_file)
    account_url = result.groups()[0]
    container_name = result.groups()[1]
    blob_name = result.groups()[2]

    with BlobClient(
        account_url,
        container_name,
        blob_name,
        max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
        max_single_get_size=1024 * 1024 * 32,  # 32 MiB
    ) as blob_client:
        local_file_path = test_dir / remote_file_name
        if check_azure_remote_file_matching_local(blob_client, local_file_path):
            print(f"  Skipping '{remote_file_name}' download (local MD5 hash matches)")
            return

        print(f"  Downloading '{remote_file_name}' to '{relative_dir}'")
        with open(local_file_path, mode="wb") as local_blob:
            download_stream = blob_client.download_blob(max_concurrency=4)
            local_blob.write(download_stream.readall())


def download_generic_remote_file(test_dir: Path, remote_file: str):
    # TODO(scotttodd): use https://pypi.org/project/requests/
    raise NotImplementedError("generic remote file downloads not implemented yet")


def download_for_test_case(test_dir: Path, test_case_json: dict):
    # This is naive (greedy, serial) for now. We could batch downloads that
    # share a source:
    #   * Iterate over all files (across all included paths), building a list
    #     of files to download (checking hashes / local references before
    #     adding to the list)
    #   * (Optionally) Determine disk space needed/available and ask before
    #     continuing
    #   * Group files based on source (e.g. Azure container)
    #   * Start batched/parallel downloads

    if "remote_files" not in test_case_json:
        return

    for remote_file in test_case_json["remote_files"]:
        if "blob.core.windows.net" in remote_file:
            download_azure_remote_file(test_dir, remote_file)
        else:
            download_generic_remote_file(test_dir, remote_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote file downloader.")
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory to search for files to download from (e.g. 'pytorch/models/resnet50')",
    )
    args = parser.parse_args()

    for test_cases_path in (THIS_DIR / args.root_dir).rglob("*.json"):
        with open(test_cases_path) as f:
            test_cases_json = pyjson5.load(f)
            if test_cases_json.get("file_format", "") != "test_cases_v0":
                continue

            print(f"Processing {test_cases_path.relative_to(THIS_DIR)}")

            test_dir = test_cases_path.parent
            for test_case_json in test_cases_json["test_cases"]:
                download_for_test_case(test_dir, test_case_json)
