# Geronimo
An intuitive, fast and lightweight machine-learning library for C++, inspired by the Keras API.

I made this primarily for educational purposes, but also for the lack of an intuitive, open-source C++ API. Python is a joke.

If you'd be interested in contributing to this project, or have any suggestions I'd love to hear from you!

## Dependencies
For all the scary linear algebra (required):
- [Armadillo C++ Matrix Library](https://gitlab.com/conradsnicta/armadillo-code)

Additionally, for data visualization (optional):
- [Matplot++ Library](https://alandefreitas.github.io/matplotplusplus/)

## Building
This project is currently built using Make. Note that I may add alternative build systems in the future.

Run `make TEST_FILENAMES=[name of test file]` in the root project directory.

Example: `make TEST_FILENAMES=example_xor` will build using the **/test/example_xor.cpp** source file.
