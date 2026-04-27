import os
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

import requests

from veritas import database, ezmm_path, nextcloud

db_database, db_user, db_password, db_host, db_port = database.values()
nextcloud_url, nextcloud_user, nextcloud_password, nextcloud_backup_dir = nextcloud.values()

# Temporary local storage for the backup files
TEMP_BACKUP_DIR = Path("temp/backup")
# Size of archive files (in bytes, Nextcloud allows 5GB max)
BATCH_SIZE_THRESHOLD = 5 * 1000 * 1000 * 1000

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
    """Uploads a file to Nextcloud using WebDAV."""
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

    # Construct the WebDAV URL
    base_dav_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{nextcloud_user}"
    target_url = f"{base_dav_url}{nextcloud_backup_dir.rstrip('/')}/{remote_file_path}"

    try:
        with open(local_file_path, "rb") as f:
            response = requests.put(
                target_url,
                data=f,
                auth=(nextcloud_user, nextcloud_password),
                verify=True
            )

        if response.status_code in [201, 204]:
            print(f"Successfully uploaded {remote_file_path} to Nextcloud.")
            return True
        else:
            print(f"Failed to upload {remote_file_path}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error uploading to Nextcloud: {e}")
        return False


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
    if backup_db and create_postgres_dump(db_dump_file):
        # Upload Postgres backup
        remote_file_path = f"{timestamp}/{file_name}"
        if not upload_to_nextcloud(db_dump_file, remote_file_path):
            overall_success = False
        # Cleanup DB dump immediately
        if os.path.exists(db_dump_file):
            os.remove(db_dump_file)
    else:
        overall_success = False

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


if __name__ == "__main__":
    main(backup_splits=False, backup_db=True, backup_ezmm=True)
