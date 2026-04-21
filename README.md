
for install of gym-pybullet-drones on windows side:

1. verify that you have python in cmd/powershell

	python --version

be in wanted project base-directory:
2. Create Virtual Environment:

	python -m venv drones-env
	drones-env\Scripts\activate

3. Upgrade packaging tools

	python -m pip install --upgrade pip setuptools wheel

4. Clone the repository

	git clone https://github.com/utiasDSL/gym-pybullet-drones.git


6. Install dependencies

	cd gym-pybullet-drones

	pip install numpy scipy matplotlib pandas

	pip install pybullet gym

8. Install project

	pip install -e .
	
	
9. Test the install

	python example\fly.py
or	python example\learn.py
	

