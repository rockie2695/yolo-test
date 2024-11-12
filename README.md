mkdir myproject
cd myproject
py -3 -m venv .venv

python -m venv --upgrade venv

.venv\Scripts\activate

pip install -r requirements.txt

python main.py