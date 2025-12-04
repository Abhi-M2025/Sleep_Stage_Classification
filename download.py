import requests
import os

BASE = "https://physionet.org/files/hmc-sleep-staging/1.1/recordings"
def download(url, output):
    print("Downloading:", output)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(output, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

os.makedirs("raw", exist_ok=True)

for i in range(65, 72):
    sn = f"SN{i:03d}"

    # recording
    url1 = f"{BASE}/{sn}.edf?download"
    out1 = f"raw/{sn}.edf"

    # scoring
    url2 = f"{BASE}/{sn}_sleepscoring.edf?download"
    out2 = f"raw/{sn}_sleepscoring.edf"

    download(url1, out1)
    download(url2, out2)