Sleep_Stage_Classification

Steps:

1. Import requirements in requirements.txt file
- pip3 install -r requirements.txt
2. If you want to download any of the edf files labeled SNXXX, use the following curl commands:
  curl -L -o SN002.edf "https://physionet.org/files/hmc-sleep-staging/1.1/recordings/SN002.edf?download"
  curl -L -o SN002_sleepscoring.edf "https://physionet.org/files/hmc-sleep-staclearging/1.1/recordings/SN002_sleepscoring.edf?download". Or, you can also use the download.py file,
  modify the bounds of the "for i in range" loop, and it will retrieve and download the corresponding edf files for those recordings.
  - Right now, the project uses SN001, SN002, SN003 to train the model and SN005 to test, but we will have to use more to train it fully.
