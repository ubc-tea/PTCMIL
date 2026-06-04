import html.parser
import urllib.parse
import urllib.request
from pathlib import Path


FILES = [
    {
        "name": "normal_001.tif",
        "id": "0BzsdkU4jWx9BLVNUUzk4dUxHWHM",
        "resourcekey": "0-DGWsjN2D_BgbgPSmaVoGkA",
    },
    {
        "name": "tumor_038.tif",
        "id": "0BzsdkU4jWx9BN3pqYkQzdlZrekk",
        "resourcekey": "0-qiEUcUKtgJwgi97DCKnXqw",
    },
]


class DriveConfirmFormParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.inputs = {}
        self._in_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self._in_form = True
            self.action = attrs.get("action")
        elif self._in_form and tag == "input" and attrs.get("name"):
            self.inputs[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self._in_form = False


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})


def resolve_download_url(file_id, resource_key):
    url = (
        "https://drive.google.com/uc?"
        + urllib.parse.urlencode(
            {"export": "download", "id": file_id, "resourcekey": resource_key}
        )
    )
    with urllib.request.urlopen(request(url), timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()
        body = response.read()

    if "text/html" not in content_type:
        return final_url

    parser = DriveConfirmFormParser()
    parser.feed(body.decode("utf-8", "ignore"))
    if not parser.action:
        raise RuntimeError("Could not find Google Drive confirmation form.")
    return parser.action + "?" + urllib.parse.urlencode(parser.inputs)


def download(file_info, out_dir):
    out_path = out_dir / file_info["name"]
    if out_path.exists():
        print(f"{file_info['name']}: already exists")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    url = resolve_download_url(file_info["id"], file_info["resourcekey"])
    print(f"{file_info['name']}: downloading")

    with urllib.request.urlopen(request(url), timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with tmp_path.open("wb") as handle:
            done = 0
            next_report = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total and done >= next_report:
                    print(
                        f"{file_info['name']}: {done / total:.1%} "
                        f"({done}/{total})",
                        flush=True,
                    )
                    next_report += max(total // 20, 1)
    tmp_path.rename(out_path)
    print(f"{file_info['name']}: saved to {out_path}")


def main():
    out_dir = Path(__file__).resolve().parent / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_info in FILES:
        download(file_info, out_dir)


if __name__ == "__main__":
    main()
