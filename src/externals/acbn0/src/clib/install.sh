#!/usr/bin/sh
# Commands to compile and install the cints.so shared library from PyQuante
# Replace your own $PATH to the include/python2.7 directory

gcc -c -fPIC cints.c -o cints.o -I/home/marco/anaconda2/include/python2.7
gcc -shared -o cints.so -fPIC cints.c -I/home/marco/anaconda2/include/python2.7
mv cints.so ../
rm cints.o
