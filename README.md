# CORGIDS

-----The following resource and CORGIDS source code was developped by Ekta Aggarwal----

## Steps to use CORGIDS:
(In the steps below "root-dir" refers to the source directory of the CORGIDS local copy)

1. Install hmm-learn library.
```bash
pip install hmmlearn
```
2. Clone the CORGIDS repository.
```bash
git clone https://github.com/ektaaggarwal24/CORGIDS.git
```

PLATFORM below refers to value which can be either "UAV" or "SAP". Based on this argument, the IDS will be generated either for a UAV or a SAP
3. Generating Intrusion detection model:
   - Run 
   ```bash
   python <root-dir>/IDSGenerator/IDSGenerator.py <PLATFORM>
   ```
   
4. Detecting Intrusion:
   - Run
   ```bash
   python <root-dir>/IntrusionDetector/IntrusionDetector.py <PLATFORM>
   ```
 
