#!/usr/bin/env python

"""
Encrypt or decrypt a directory using Fernet symmetric encryption.

This script processes .md, .txt, and .py files in a directory, encrypting or
decrypting them using the VIM_SECRET environment variable.

Examples:
# Encrypt a directory (creates DIR.secret and removes DIR)
> encrypt.py --input_dir data605/lectures_quizzes

# Encrypt a directory but keep the original
> encrypt.py --input_dir data605/lectures_quizzes --keep_old_dir

# Decrypt a directory (creates DIR from DIR.secret)
> encrypt.py --input_dir data605/lectures_quizzes --decrypt

Import as:

import encrypt as encrypt
"""

import argparse
import base64
import hashlib
import logging
import os
import shlex
import shutil
from pathlib import Path

from cryptography.fernet import Fernet

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hsystem as hsystem

_LOG = logging.getLogger(__name__)

# Valid file extensions to process.
_VALID_EXTENSIONS = (".md", ".txt", ".py")

# Encryption header to identify encrypted files.
_ENCRYPTION_HEADER = b"FERNET_ENCRYPTED_V1\n"

# #############################################################################


def _get_secret() -> str:
    """
    Get the VIM_SECRET environment variable.

    :return: the VIM_SECRET value
    """
    secret = os.environ.get("VIM_SECRET")
    hdbg.dassert(
        secret,
        "VIM_SECRET environment variable is not set or is empty",
    )
    return secret


def _derive_key(password: str) -> bytes:
    """
    Derive a Fernet-compatible key from a password.

    Uses SHA256 to derive a 32-byte key and base64 encodes it for Fernet.

    :param password: the password to derive key from
    :return: base64-encoded 32-byte key suitable for Fernet
    """
    # Hash the password to get 32 bytes.
    key_bytes = hashlib.sha256(password.encode()).digest()
    # Base64 encode for Fernet.
    return base64.urlsafe_b64encode(key_bytes)


def _should_process_file(file_path: Path) -> bool:
    """
    Check if a file should be processed based on its extension.

    :param file_path: path to the file
    :return: True if file has a valid extension
    """
    return file_path.suffix in _VALID_EXTENSIONS


def _is_file_encrypted(file_path: Path) -> bool:
    """
    Check if a file is already encrypted by looking for encryption header.

    :param file_path: path to the file
    :return: True if file starts with encryption header
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(len(_ENCRYPTION_HEADER))
            return header == _ENCRYPTION_HEADER
    except Exception as e:
        _LOG.warning("Could not read file %s: %s", file_path, e)
        return False


def _encrypt_file(file_path: Path, secret: str) -> None:
    """
    Encrypt a single file using Fernet encryption.

    Skips files that are already encrypted.

    :param file_path: path to the file to encrypt
    :param secret: encryption key
    """
    # Check if file is already encrypted.
    if _is_file_encrypted(file_path):
        _LOG.debug("Skipping already encrypted file: %s", file_path)
        return
    _LOG.debug("Encrypting file: %s", file_path)
    # Read the file content.
    with open(file_path, "rb") as f:
        plaintext = f.read()
    # Derive encryption key.
    key = _derive_key(secret)
    cipher = Fernet(key)
    # Encrypt the content.
    encrypted = cipher.encrypt(plaintext)
    # Write encrypted content with header.
    with open(file_path, "wb") as f:
        f.write(_ENCRYPTION_HEADER)
        f.write(encrypted)


def _decrypt_file(file_path: Path, secret: str) -> None:
    """
    Decrypt a single file using Fernet decryption.

    :param file_path: path to the file to decrypt
    :param secret: decryption key
    """
    _LOG.debug("Decrypting file: %s", file_path)
    # Read the encrypted file.
    with open(file_path, "rb") as f:
        content = f.read()
    # Check and remove header.
    hdbg.dassert(
        content.startswith(_ENCRYPTION_HEADER),
        "File is not encrypted with expected format:",
        file_path,
    )
    encrypted = content[len(_ENCRYPTION_HEADER) :]
    # Derive decryption key.
    key = _derive_key(secret)
    cipher = Fernet(key)
    # Decrypt the content.
    try:
        plaintext = cipher.decrypt(encrypted)
    except Exception as e:
        hdbg.dfatal(
            "Failed to decrypt file (wrong password?):",
            file_path,
            str(e),
        )
    # Write decrypted content.
    with open(file_path, "wb") as f:
        f.write(plaintext)


def _make_readonly(dir_path: Path) -> None:
    """
    Make all files in a directory read-only recursively.

    :param dir_path: path to the directory
    """
    _LOG.debug("Making directory read-only: %s", dir_path)
    quoted_path = shlex.quote(str(dir_path))
    cmd = f"chmod -R -w {quoted_path}"
    hsystem.system(cmd)


def _encrypt_directory(
    *,
    input_dir: str,
    keep_old_dir: bool,
) -> None:
    """
    Encrypt a directory by creating a .secret copy with encrypted files.

    :param input_dir: source directory to encrypt
    :param keep_old_dir: if True, keep the original directory
    """
    _LOG.info("Starting encryption of directory: %s", input_dir)
    # Get the secret.
    secret = _get_secret()
    # Set up paths.
    src_path = Path(input_dir)
    hdbg.dassert(
        src_path.exists(),
        "Source directory does not exist:",
        input_dir,
    )
    hdbg.dassert(
        src_path.is_dir(),
        "Source path is not a directory:",
        input_dir,
    )
    dst_path = Path(f"{input_dir}.secret")
    # Remove existing destination if it exists.
    if dst_path.exists():
        _LOG.info("Removing existing destination: %s", dst_path)
        # Make files writable before deletion in case they were read-only.
        quoted_dst_path = shlex.quote(str(dst_path))
        cmd = f"chmod -R +w {quoted_dst_path}"
        hsystem.system(cmd)
        shutil.rmtree(dst_path)
    # Copy the directory.
    _LOG.info("Copying directory to: %s", dst_path)
    shutil.copytree(src_path, dst_path)
    # Make files writable before encryption.
    quoted_path = shlex.quote(str(dst_path))
    cmd = f"chmod -R +w {quoted_path}"
    hsystem.system(cmd)
    # Find and encrypt all valid files.
    file_count = 0
    for file_path in dst_path.rglob("*"):
        if file_path.is_file() and _should_process_file(file_path):
            _encrypt_file(file_path, secret)
            file_count += 1
    _LOG.info("Encrypted %d files", file_count)
    # Make files read-only.
    _make_readonly(dst_path)
    # Remove source directory if requested.
    if not keep_old_dir:
        _LOG.info("Removing source directory: %s", src_path)
        # Make files writable before deletion in case they were read-only.
        quoted_src_path = shlex.quote(str(src_path))
        cmd = f"chmod -R +w {quoted_src_path}"
        hsystem.system(cmd)
        shutil.rmtree(src_path)
    _LOG.info("Encryption complete")


def _decrypt_directory(
    *,
    input_dir: str,
    keep_old_dir: bool,
) -> None:
    """
    Decrypt a .secret directory by creating the original directory.

    :param input_dir: base name of the directory (without .secret)
    :param keep_old_dir: if True, keep the .secret directory
    """
    _LOG.info("Starting decryption of directory: %s", input_dir)
    # Get the secret.
    secret = _get_secret()
    # Set up paths.
    src_path = Path(f"{input_dir}.secret")
    hdbg.dassert(
        src_path.exists(),
        "Source .secret directory does not exist:",
        src_path,
    )
    hdbg.dassert(
        src_path.is_dir(),
        "Source path is not a directory:",
        src_path,
    )
    dst_path = Path(input_dir)
    # Remove existing destination if it exists.
    if dst_path.exists():
        _LOG.info("Removing existing destination: %s", dst_path)
        # Make files writable before deletion in case they were read-only.
        quoted_dst_path = shlex.quote(str(dst_path))
        cmd = f"chmod -R +w {quoted_dst_path}"
        hsystem.system(cmd)
        shutil.rmtree(dst_path)
    # Copy the directory.
    _LOG.info("Copying directory to: %s", dst_path)
    shutil.copytree(src_path, dst_path)
    # Make files writable before decryption.
    quoted_path = shlex.quote(str(dst_path))
    cmd = f"chmod -R +w {quoted_path}"
    hsystem.system(cmd)
    # Find and decrypt all valid files.
    file_count = 0
    for file_path in dst_path.rglob("*"):
        if file_path.is_file() and _should_process_file(file_path):
            _decrypt_file(file_path, secret)
            file_count += 1
    _LOG.info("Decrypted %d files", file_count)
    # Remove source directory if requested.
    if not keep_old_dir:
        _LOG.info("Removing source .secret directory: %s", src_path)
        # Make files writable before deletion.
        quoted_src_path = shlex.quote(str(src_path))
        cmd = f"chmod -R +w {quoted_src_path}"
        hsystem.system(cmd)
        shutil.rmtree(src_path)
    _LOG.info("Decryption complete")


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        action="store",
        required=True,
        help="Directory to encrypt or decrypt (base name without .secret)",
    )
    parser.add_argument(
        "--keep_old_dir",
        action="store_true",
        default=False,
        help="Keep the original directory after encryption/decryption",
    )
    parser.add_argument(
        "--decrypt",
        action="store_true",
        default=False,
        help="Decrypt from .secret directory instead of encrypting",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Perform encryption or decryption.
    if args.decrypt:
        _decrypt_directory(
            input_dir=args.input_dir,
            keep_old_dir=args.keep_old_dir,
        )
    else:
        _encrypt_directory(
            input_dir=args.input_dir,
            keep_old_dir=args.keep_old_dir,
        )


if __name__ == "__main__":
    _main(_parse())
