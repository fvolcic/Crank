"""
Build and run the test suite
"""

from os import system

print("Building...")
system("g++ tests/fftests/backprop.cpp -D NN_DEBUG -g3 -o bin/backprop_tests")
system("g++ tests/fftests/fftests.cpp -g3 -o bin/fftests_test")

print("\n\nBuilding complete.")
print("Running tests...\n\n")

system("./bin/backprop_tests")
system("./bin/fftests_test")
