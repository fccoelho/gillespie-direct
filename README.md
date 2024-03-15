# gillespyx
Implementation of the [Gillespie direct method (SSA)](https://en.wikipedia.org/wiki/Gillespie_algorithm) in Cython, for simulating stochastic differential equations as a markov process.

## Building the library

You need `libgsl-dev` in order to build this package. On Ubuntu, you can install it with:

```bash
sudo apt-get install libgsl-dev
```

You also need to ha sconstruct installed on your system. You can install it with:

```bash
pip install scons
```

Then, you can build the library with:

```bash
poetry install
poetry shell
cd gillespyx
scons
```
The above command will create a `gillespix.so` file in the `gillespyx` directory.

## Usage

You can use the library as follows just copy the file `gillespyx.so` to your project directory and import it in your python code.

```python
import gillespyx
```python

