#!/usr/bin/env python
# coding: utf-8
import os
import gc
import time

class Test:
    def __init__(self):
        self.name = "haha"

def run():
    L = []
    begin = time.time()
    for i in xrange(10000000):
        L.append(Test())
    end = time.time()
    print 'cost:%s' %(round(end-begin,3))
    input_str = raw_input('\nrun: ')
if __name__=="__main__":
    print "pid:%s" %os.getpid()
    input_str = raw_input('\nstart')
    run()
    input_str = raw_input('\nready to collect')
    gc.collect()
    input_str = raw_input('\ncollected.')

