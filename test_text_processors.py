import unittest

import text_processor


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        text = ['football - champleaque grd - psv', 'eindhoven-atleticomadrid - 1/n/2', '358 1 4, 40']
        # csvfile = 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'
        events = ['Roda - PSV Eindhoven', 'PSVEindhov.-AtleticoMadr.']

        # Find most likely bets
        betstrings = text_processor.segmentBets(text)
        probableevents = [text_processor.findMostProbableEvent(events, b) for b in betstrings]
        self.assertEqual(probableevents, ['PSVEindhov.-AtleticoMadr.'])
        # import ipdb; ipdb.set_trace()
        # pass


if __name__ == '__main__':
    unittest.main()
