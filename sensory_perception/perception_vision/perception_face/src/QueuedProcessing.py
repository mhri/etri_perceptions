# coding: utf-8

'''
Queue를 매개로 쓰레드를 두고 작업을 처리하는 클래스

Author: Minsu Jang (minsu@etri.re.kr)
'''

from Queue import Queue
from Queue import Empty
from threading import Thread
import time
from abc import ABCMeta, abstractmethod

class QueuedProcessor(object):
    '''
    작업 총괄 클래스
    '''

    __metaclass__ = ABCMeta

    def __init__(self, worker):
        self.data_queue = Queue()
        self.result_queue = Queue()
        self.timestamp_last_call = time.time()
        self.worker = worker
        self.worker.set_queues(self.data_queue, self.result_queue)
        self.worker.start()

    @abstractmethod
    def processing_is_necessary(self, data):
        '''
        Check whether data processing is necessary or not.
        '''
        pass
        
    def process(self, data_id, data):
        '''
        Process a data.
        '''
        if self.processing_is_necessary(data):
            self.data_queue.put((data_id, data))
            self.timestamp_last_call = time.time()

        try:
            data_id, result = self.result_queue.get_nowait()
            self.result_queue.task_done()
        except Empty:
            result = None
            
        return data_id, result

    def stop(self):
        '''
        Stop the worker thread.
        '''
        print "Stop Called..."
        self.worker.stop()


class Worker(Thread):
    '''
    작업 수행 클래스
    '''
    def __init__(self):
        Thread.__init__(self)
        self.data_queue = None
        self.result_queue = None
        self.suspend_ = False
        self.exit_ = False

    def set_queues(self, data_queue, result_queue):
        '''
        Set queues for receiving data and returning results.
        '''
        self.data_queue = data_queue
        self.result_queue = result_queue

    def work(self, data):
        '''
        Process the data.
        '''
        return "Not Implemented"

    def run(self):
        print "Run called @Worker"
        while True:
            ### Suspend ###
            while self.suspend_:
                time.sleep(0.5)

            ### Process ###
            #print 'Thread process !!!'
            try:
                data_id, data = self.data_queue.get()
                self.data_queue.task_done()
                result = self.work(data)
                self.result_queue.put((data_id, result))
            except Empty:
                time.sleep(0.001)

            ### Exit ###
            if self.exit_:
                break

    def suspend(self):
        '''
        Suspend this thread.
        '''
        self.suspend_ = True

    def resume(self):
        '''
        Resume execution of this thread.
        '''
        self.suspend_ = False

    def stop(self):
        '''
        Stop this thread.
        '''
        print "Stop Called @Worker"
        self.exit_ = True
