from io import BytesIO

import requests


def fetch_from_url(url: str) -> BytesIO:
    """
    Fetch content from a given URL and return a BytesIO object.

    :param url: The URL to fetch.
    :return: A BytesIO object containing the fetched content.
    :raises RuntimeError: If there is an issue fetching the content.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BytesIO(response.content)


def is_url(path: str) -> bool:
    """
    Check if a given path is a URL.

    :param path: The path to check.
    :return: True if the path is a URL, False otherwise.
    """
    return path.startswith("http://") or path.startswith("https://")
