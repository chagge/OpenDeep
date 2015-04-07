'''
Unit testing for the midi datasets
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import unittest
import logging
import numpy
# internal references
from opendeep.data.standard_datasets.midi.musedata import MuseData
from opendeep.data.standard_datasets.midi.jsb_chorales import JSBChorales
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.data.standard_datasets.midi.piano_midi_de import PianoMidiDe
from opendeep.data.dataset import TRAIN, VALID, TEST
import opendeep.log.logger as logger
from opendeep.data.iterators.sequential import SequentialIterator
from opendeep.data.iterators.random import RandomIterator


class TestMuse(unittest.TestCase):

    def setUp(self):
        print "setting up!"
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the muse dataset
        self.muse = MuseData()
        # get the jsb dataset
        # self.jsb = JSBChorales()
        # get nottingham dataset
        self.nottingham = Nottingham()
        # get the piano-midi-de dataset
        # self.piano = PianoMidiDe()


    def testSizes(self):
        print 'muse train', self.muse.train.shape.eval()[0]
        assert self.muse.train.shape.eval()[0] == numpy.sum([l[0] for l in self.muse.getDataShape(TRAIN)])
        assert self.muse.valid.shape.eval()[0] == numpy.sum([l[0] for l in self.muse.getDataShape(VALID)])
        assert self.muse.test.shape.eval()[0]  == numpy.sum([l[0] for l in self.muse.getDataShape(TEST)])

        print 'nottingham train', self.nottingham.train.shape.eval()[0]
        assert self.nottingham.train.shape.eval()[0] == numpy.sum([l[0] for l in self.nottingham.getDataShape(TRAIN)])
        assert self.nottingham.valid.shape.eval()[0] == numpy.sum([l[0] for l in self.nottingham.getDataShape(VALID)])
        assert self.nottingham.test.shape.eval()[0] == numpy.sum([l[0] for l in self.nottingham.getDataShape(TEST)])

    def testSequentialIterator(self):
        self.log.debug('TESTING MUSE SEQUENTIAL ITERATOR')
        i = 0
        for _, y in SequentialIterator(dataset=self.muse, batch_size=100, minimum_batch_size=1, subset=TRAIN):
            i += 1
        print i

        i = 0
        for _, y in SequentialIterator(dataset=self.muse, batch_size=100, minimum_batch_size=100, subset=TRAIN):
            i += 1
        print i

        self.log.debug('TESTING NOTTINGHAM SEQUENTIAL ITERATOR')
        i = 0
        for _, y in SequentialIterator(dataset=self.nottingham, batch_size=100, minimum_batch_size=1, subset=TRAIN):
            i += 1
        print i

        i = 0
        for _, y in SequentialIterator(dataset=self.nottingham, batch_size=100, minimum_batch_size=100, subset=TRAIN):
            i += 1
        print i


    def tearDown(self):
        del self.muse
        # del self.jsb
        del self.nottingham
        # del self.piano
        print "done!"


if __name__ == '__main__':
    unittest.main()