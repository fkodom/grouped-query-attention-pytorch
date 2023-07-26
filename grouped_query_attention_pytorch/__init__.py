import os
from subprocess import getoutput


def get_version_tag() -> str:
    try:
        env_key = "GROUPED_QUERY_ATTENTION_PYTORCH_VERSION".upper()
        version = os.environ[env_key]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    if version.lower().startswith("fatal"):
        version = "0.0.0"

    return version


VERSION = get_version_tag()
