mkdir myproject
cd myproject
py -3 -m venv .venv

python -m venv --upgrade venv

.venv\Scripts\activate

pip install -r requirements.txt

## run
python detect.py

python segment.py

python classify.py

python detect.py

python obb.py

python pose.py
