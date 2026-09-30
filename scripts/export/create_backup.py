import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import requests

from veritas import database, ezmm_path, nextcloud

db_database, db_user, db_password, db_host, db_port = database.values()
# Read by key, so that the section can gain settings without breaking this script.
nextcloud_url = nextcloud.get("url")
nextcloud_user = nextcloud.get("user")
nextcloud_password = nextcloud.get("password")
nextcloud_backup_dir = nextcloud.get("backup_dir")

# Temporary local storage for the backup files
TEMP_BACKUP_DIR = Path("temp/backup")

#: Largest file uploaded in one request (bytes). A larger file is uploaded in parts
#: of at most this size - `<name>.part001`, `<name>.part002`, ... - together with a
#: manifest `<name>.parts.json` that says how to reassemble it. Configurable as
#: `nextcloud.max_file_size` in config.yaml.
MAX_FILE_SIZE = int(nextcloud.get("max_file_size") or 5 * 1000 * 1000 * 1000)
#: If the server - or the proxy in front of it, which is where "413 Request Entity
#: Too Large" usually comes from - rejects a part as too large, the part size is
#: halved and the upload retried, but never below this.
MIN_PART_SIZE = 16 * 1000 * 1000
#: Size of the ezMM archive batches; kept at the upload limit, so that a batch is
#: normally uploaded in one piece.
BATCH_SIZE_THRESHOLD = MAX_FILE_SIZE
#: How much of a file is read at a time while hashing or uploading it.
READ_BLOCK = 8 * 1024 * 1024

HTTP_TOO_LARGE = 413

# TODO: Upload only ezmm media that occur in the DB data

def create_postgres_dump(output_path):
    """Creates a dump of the Postgres database."""
    print(f"Creating Postgres dump for database '{db_database}'...")

    # Set the PGPASSWORD environment variable to avoid interactive password prompt
    env = os.environ.copy()
    env["PGPASSWORD"] = db_password

    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "pg_dump",
        "-h", db_host,
        "-p", str(db_port),
        "-U", db_user,
        "-Fc",  # Custom format (compressed)
        "-f", output_path,
        db_database
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Postgres dump created at {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating Postgres dump: {e}")
        return False


def ensure_remote_dir(remote_dir_path: str):
    """Ensures that the remote directory structure exists on Nextcloud."""
    base_dav_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{nextcloud_user}"
    full_backup_dir = f"{nextcloud_backup_dir.strip('/')}/{remote_dir_path.strip('/')}".strip('/')
    
    parts = full_backup_dir.split('/')
    current_path = ""
    
    for part in parts:
        if not part:
            continue
        current_path = f"{current_path}/{part}"
        url = f"{base_dav_url}{current_path}/"
        
        # Check if directory exists
        response = requests.request(
            "PROPFIND",
            url,
            auth=(nextcloud_user, nextcloud_password),
            verify=True,
            headers={"Depth": "0"}
        )
        
        if response.status_code == 404:
            print(f"Creating remote directory: {current_path}")
            mkcol_response = requests.request(
                "MKCOL",
                url,
                auth=(nextcloud_user, nextcloud_password),
                verify=True
            )
            if mkcol_response.status_code not in [201, 405]:  # 405 means it might have been created meanwhile
                print(f"Failed to create directory {current_path}. Status: {mkcol_response.status_code}")
                return False
        elif response.status_code not in [200, 207]:
            print(f"Error checking directory {current_path}. Status: {response.status_code}")
            return False
            
    return True


def remote_file_exists(remote_file_path: str):
    """Checks if a file exists on Nextcloud using WebDAV PROPFIND."""
    base_dav_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{nextcloud_user}"
    target_url = f"{base_dav_url}{nextcloud_backup_dir.rstrip('/')}/{remote_file_path.lstrip('/')}"
    
    try:
        response = requests.request(
            "PROPFIND",
            target_url,
            auth=(nextcloud_user, nextcloud_password),
            verify=True,
            headers={"Depth": "0"}
        )
        return response.status_code in [200, 207]
    except Exception as e:
        print(f"Error checking if {remote_file_path} exists: {e}")
        return False


def upload_to_nextcloud(local_file_path, remote_file_path: str, skip_if_exists: bool = False):
    """Uploads a file to Nextcloud using WebDAV - in parts, if it is larger than
    `MAX_FILE_SIZE` (see `upload_file`)."""
    if skip_if_exists and remote_file_exists(remote_file_path):
        print(f"File {remote_file_path} already exists on Nextcloud. Skipping upload.")
        return True

    print(f"Uploading {remote_file_path} to Nextcloud...")

    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote_file_path)
    if remote_dir:
        if not ensure_remote_dir(remote_dir):
            print(f"Could not ensure remote directory {remote_dir} exists.")
            return False

    return upload_file(local_file_path, remote_file_path)


class FileSlice:
    """`length` bytes of a file, starting at `offset`, as a readable object with a
    length. `requests` then sends a `Content-Length` and streams the slice from
    disk, so a part is never copied into memory or into a temporary file."""

    def __init__(self, path, offset: int, length: int):
        self._file = open(path, "rb")
        self._file.seek(offset)
        self._remaining = length
        self._length = length

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
        data = self._file.read(size)
        self._remaining -= len(data)
        return data

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def plan_parts(size: int, part_size: int) -> list[tuple[int, int]]:
    """`(offset, length)` of every part of a file of `size` bytes."""
    if size <= 0:
        return [(0, 0)]
    return [(offset, min(part_size, size - offset)) for offset in range(0, size, part_size)]


def part_names(remote_file_path: str, n_parts: int) -> list[str]:
    """`<name>.part001`, ... - zero-padded, so that the parts sort in order and
    `cat <name>.part*` reassembles them."""
    width = max(3, len(str(n_parts)))
    return [f"{remote_file_path}.part{index:0{width}d}" for index in range(1, n_parts + 1)]


def sha256_of(path, offset: int = 0, length: int | None = None) -> str:
    """SHA-256 of the whole file, or of `length` bytes from `offset`."""
    digest = hashlib.sha256()
    with FileSlice(path, offset, os.path.getsize(path) - offset if length is None else length) as f:
        while block := f.read(READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(local_file_path, remote_file_path: str, parts: list[tuple[int, int]],
                   names: list[str]) -> dict:
    """What it takes to reassemble and verify a file uploaded in parts."""
    file_name = os.path.basename(remote_file_path)
    return {
        "file": file_name,
        "size": os.path.getsize(local_file_path),
        "sha256": sha256_of(local_file_path),
        "part_size": parts[0][1],
        "parts": [{"name": os.path.basename(name), "size": length,
                   "sha256": sha256_of(local_file_path, offset, length)}
                  for (offset, length), name in zip(parts, names)],
        "restore": f"cat {file_name}.part* > {file_name} && sha256sum {file_name}",
    }


def _target_url(remote_file_path: str) -> str:
    base_dav_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{nextcloud_user}"
    return f"{base_dav_url}{nextcloud_backup_dir.rstrip('/')}/{remote_file_path.lstrip('/')}"


def _put(remote_file_path: str, data) -> int | None:
    """Uploads `data` (bytes or a `FileSlice`) to the path. Returns the HTTP status,
    or None if the request failed without one."""
    try:
        response = requests.put(_target_url(remote_file_path), data=data,
                                auth=(nextcloud_user, nextcloud_password), verify=True)
    except Exception as e:
        print(f"Error uploading {remote_file_path} to Nextcloud: {e}")
        return None
    if response.status_code not in (201, 204):
        print(f"Failed to upload {remote_file_path}. Status code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
    return response.status_code


def _succeeded(status: int | None) -> bool:
    return status in (201, 204)


def upload_file(local_file_path, remote_file_path: str, part_size: int | None = None) -> bool:
    """Uploads a file in one request if it fits into `part_size` (default
    `MAX_FILE_SIZE`), in parts plus a manifest otherwise.

    A "413 Request Entity Too Large" means the limit is lower than assumed - often
    a proxy's, not Nextcloud's - so the part size is halved and the upload
    retried, down to `MIN_PART_SIZE`. Parts are overwritten on a retry; since a
    smaller part size only ever produces more parts, none of an earlier attempt
    is left behind under a name the manifest does not list."""
    size = os.path.getsize(local_file_path)
    part_size = part_size or MAX_FILE_SIZE

    while True:
        if size <= part_size:
            with FileSlice(local_file_path, 0, size) as data:
                status = _put(remote_file_path, data)
        else:
            status = _upload_parts(local_file_path, remote_file_path, size, part_size)

        if _succeeded(status):
            print(f"Successfully uploaded {remote_file_path} to Nextcloud.")
            return True
        if status != HTTP_TOO_LARGE or part_size <= MIN_PART_SIZE:
            return False

        part_size = max(MIN_PART_SIZE, min(part_size, size) // 2)
        print(f"{remote_file_path} is too large for the server; retrying in parts of "
              f"{part_size / 1e9:.2f} GB.")


def _upload_parts(local_file_path, remote_file_path: str, size: int,
                  part_size: int) -> int | None:
    """Uploads the parts and then the manifest. Returns the status of the first
    failed request, or that of the manifest upload if all parts succeeded."""
    parts = plan_parts(size, part_size)
    names = part_names(remote_file_path, len(parts))
    print(f"Uploading {remote_file_path} in {len(parts)} parts of up to "
          f"{part_size / 1e9:.2f} GB...")
    for (offset, length), name in zip(parts, names):
        with FileSlice(local_file_path, offset, length) as data:
            status = _put(name, data)
        if not _succeeded(status):
            return status
        print(f"  uploaded {os.path.basename(name)}")

    manifest = build_manifest(local_file_path, remote_file_path, parts, names)
    return _put(f"{remote_file_path}.parts.json",
                json.dumps(manifest, indent=2).encode("utf-8"))


def backup_ezmm_registry(timestamp: str):
    """Backs up the ezMM registry in ~100 GB batches."""
    print(f"Starting batched backup for ezMM registry from '{ezmm_path}'...")

    if not os.path.exists(ezmm_path):
        print(f"Error: ezMM path '{ezmm_path}' does not exist.")
        return False

    backup_dir = TEMP_BACKUP_DIR / timestamp
    target_dir = backup_dir / "ezmm"
    target_dir.mkdir(parents=True, exist_ok=True)

    success = True
    batch_index = 1
    current_batch_files = []
    current_batch_size = 0

    def process_batch(files, index):
        nonlocal success
        archive_path = target_dir / f"{index}.tar"
        
        print(f"Creating batch {index} archive ({len(files)} files)...")
        try:
            with tarfile.open(archive_path, "w") as tar:
                for file_path in files:
                    # Use relative path in archive
                    arcname = os.path.relpath(file_path, ezmm_path)
                    tar.add(file_path, arcname=arcname)

            remote_archive_path = f"{timestamp}/ezmm/{index}.tar"
            if upload_to_nextcloud(archive_path, remote_archive_path):
                print(f"Successfully backed up batch {index}.")
            else:
                print(f"Failed to upload batch {index}.")
                success = False
            
            if os.path.exists(archive_path):
                os.remove(archive_path)
                
        except Exception as e:
            print(f"Error processing batch {index}: {e}")
            success = False

    # Walk through the registry and collect files
    for root, dirs, files in os.walk(ezmm_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                
                # If a single file is larger than the threshold, it will be its own batch
                if current_batch_size + file_size > BATCH_SIZE_THRESHOLD and current_batch_files:
                    process_batch(current_batch_files, batch_index)
                    batch_index += 1
                    current_batch_files = []
                    current_batch_size = 0
                
                current_batch_files.append(file_path)
                current_batch_size += file_size
                
                # If it's already over threshold with this one file, process it (if we haven't just cleared)
                if current_batch_size >= BATCH_SIZE_THRESHOLD:
                    process_batch(current_batch_files, batch_index)
                    batch_index += 1
                    current_batch_files = []
                    current_batch_size = 0
                    
            except Exception as e:
                print(f"Error accessing file {file_path}: {e}")
                # We continue with other files but mark as not fully successful
                success = False

    # Process the last remaining batch
    if current_batch_files:
        process_batch(current_batch_files, batch_index)

    return success


def backup_release_splits():
    """Backs up release splits from /exports to 'Release Splits' folder on Nextcloud."""
    print("Starting backup of release splits...")
    exports_dir = Path("exports")
    if not exports_dir.exists():
        print("Exports directory not found. Skipping release splits backup.")
        return True

    success = True
    # Walk through exports and find files
    for root, _, files in os.walk(exports_dir):
        for file in files:
            local_path = Path(root) / file
            # Construct remote path relative to 'Release Splits'
            # We want it to be "Release Splits/relative/path/to/file"
            relative_path = local_path.relative_to(exports_dir)
            remote_path = f"Release Splits/{relative_path.as_posix()}"
            
            if not upload_to_nextcloud(local_path, remote_path, skip_if_exists=True):
                success = False
    
    return success


def main(backup_db: bool = True,
         backup_ezmm: bool = True,
         backup_splits: bool = True):
    # Initiate temporary backup directory
    TEMP_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    backup_dir = TEMP_BACKUP_DIR / timestamp

    overall_success = True

    # 1. Backup Release Splits
    if backup_splits:
        if not backup_release_splits():
            overall_success = False

    # 2. Backup Postgres
    file_name = f"db.sql"
    db_dump_file = backup_dir / file_name
    if backup_db:
        if create_postgres_dump(db_dump_file):
            # Upload Postgres backup (in parts if it exceeds MAX_FILE_SIZE)
            remote_file_path = f"{timestamp}/{file_name}"
            if not upload_to_nextcloud(db_dump_file, remote_file_path):
                overall_success = False
        else:
            overall_success = False
        # Cleanup DB dump immediately
        if os.path.exists(db_dump_file):
            os.remove(db_dump_file)

    # 3. Backup ezMM (iterative)
    if backup_ezmm:
        if not backup_ezmm_registry(timestamp):
            overall_success = False

    # Final Cleanup of temporary backup directory
    if os.path.exists(backup_dir) and not os.listdir(backup_dir):
        os.rmdir(backup_dir)

    if overall_success:
        print("Backup process completed successfully.")
    else:
        print("Backup process failed at one or more steps.")
    return overall_success


if __name__ == "__main__":
    # A non-zero exit code, so that a scheduler notices a failed backup.
    sys.exit(0 if main(backup_splits=False, backup_db=True, backup_ezmm=False) else 1)
