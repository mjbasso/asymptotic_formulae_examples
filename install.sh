#!/bin/bash

# Create our python environment
deactivate > /dev/null 2>&1
ENV_NAME=".env"
rm -r $ENV_NAME setup.sh || true
python3 -m venv --without-pip $ENV_NAME
source $ENV_NAME/bin/activate

# Get a nice clean version of pip
curl https://bootstrap.pypa.io/get-pip.py | python3

# Enumerate our install requirements
touch requirements.txt
echo "iminuit    ~= 1.4"  >> requirements.txt
echo "matplotlib ~= 2.1"  >> requirements.txt
echo "numpy      ~= 1.19" >> requirements.txt
echo "probfit    ~= 1.1"  >> requirements.txt
echo "scipy      ~= 1.5"  >> requirements.txt

# Extra packages for running on a raspberry pi
RASPERRY_PI=true
if $RASPERRY_PI; then
    echo "cairocffi  ~= 1.2"    >> requirements.txt
    echo "pgi        ~= 0.0.11" >> requirements.txt
    echo "pycairo    ~= 1.16"   >> requirements.txt
fi

# Install our requirements
python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel
python3 -m pip install -r requirements.txt
rm requirements.txt

dpkg -s libatlas-base-dev
if [ $? -ne 0 ]; then
    echo "Warning : libatlas-base-dev is not installed and is needed for numpy"
    echo "To install, run: sudo apt-get install libatlas-base-dev"
fi

# Create a setup script for returning
SETUP_SCRIPT="setup.sh"
[ -e $SETUP_SCRIPT ] && rm $SETUP_SCRIPT
touch $SETUP_SCRIPT
echo "#!/bin/bash"                   >> $SETUP_SCRIPT
echo ""                              >> $SETUP_SCRIPT
echo "deactivate > /dev/null 2>&1"   >> $SETUP_SCRIPT
echo "source $ENV_NAME/bin/activate" >> $SETUP_SCRIPT
chmod +x $SETUP_SCRIPT

# Possibly handy printout
echo "*** Python 3 version   : $(python3 --version)"
echo "*** Pip version        : $(python3 -m pip --version)"
