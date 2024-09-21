import unittest
import os
import json
import numpy as np
import pickle

TC_PARSE_LOC = os.environ.get('TCPARSE_LOC', False)
if TC_PARSE_LOC:
    import sys
    sys.path.insert(1, TC_PARSE_LOC)

from tcparse import TCParser
    

#   turn on to print which keys and frames are being checked
PRINT_ASSERT = True

#   ignore these keys, usefull when debugging specific parsers
IGNORE_KEYS = []

def _recursive_compare_no_trace(tester: unittest.TestCase, tst_obj, ref_obj):
    '''
        Recursively iterate through a test object and compare it's members to the reference object
    '''
    tester.assertIs(type(tst_obj), type(ref_obj))
    if any(isinstance(tst_obj, x) for x in [type(None), int, str]):
        tester.assertEqual(tst_obj, ref_obj)
    elif isinstance(tst_obj, float):
        tester.assertAlmostEqual(tst_obj, ref_obj, 9)
    elif isinstance(tst_obj, np.ndarray):
        if tst_obj.dtype == float:
            np.testing.assert_almost_equal(tst_obj, ref_obj, 13)
        else:
            np.testing.assert_equal(tst_obj, ref_obj)
    elif isinstance(tst_obj, list) or isinstance(tst_obj, tuple):
        for n in range(len(tst_obj)):
            _recursive_compare_no_trace(tester, tst_obj[n], ref_obj[n])
    elif isinstance(tst_obj, dict):
        for key in tst_obj:
            _recursive_compare_no_trace(tester, tst_obj[key], ref_obj[key])
    elif '__dict__' in dir(tst_obj):
        for key in tst_obj.__dict__:
            _recursive_compare_no_trace(tester, tst_obj.__dict__[key], ref_obj.__dict__[key])
    else:
        print("COULD NOT COMPARE ", type(tst_obj))

class TestParser(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._orig_dir = os.path.abspath(os.path.curdir)

    def _test_MD_data(self, ref_data, tst_data):
        #   make sure the same frame data is present
        ref_frames = list(ref_data.keys())
        tst_frames = list(tst_data.keys())
        self.assertListEqual(ref_frames, tst_frames)
        
        #   check each frame
        for frame in ref_frames:
            if PRINT_ASSERT: print("ASSERTING FRAME: ", frame)
            ref_job = ref_data[frame]
            tst_job = tst_data[frame]

            for key in IGNORE_KEYS:
                if key in ref_job: ref_job.pop(key)
                if key in tst_job: tst_job.pop(key)

            ref_keys = set(ref_job.keys())
            tst_keys = set(tst_job.keys())

            self.assertSetEqual(ref_keys, tst_keys)


            #   check each object in the job
            for key in ref_keys:
                # if PRINT_ASSERT: print(" - ASSERTING KEY: ", key)
                # print(f"    {key} REFERENCE: ", len(ref_job[key]))
                # print(f"    {key} TEST: ",      len(tst_job[key]))
                self.assertAlmostEqual(ref_job[key], tst_job[key], places=14)


    def _test_single_job(self, ref_job, tst_job):
        
        ref_keys = list(ref_job.keys())
        tst_keys = list(tst_job.keys())

        #   check each object in the job
        for key in ref_keys:
            if key not in tst_keys:
                raise KeyError(f"Key {key} not found in test data")

            tst_val = tst_job[key]
            ref_val = ref_job[key]
            self.assertIs(type(tst_val), type(ref_val))
            if 'esp' in key:
                continue
            _recursive_compare_no_trace(self, tst_job[key], ref_job[key])

        for key in tst_job:
            if key not in ref_keys:
                raise KeyError(f"Key {key} not found in reference data")



    def test_MD(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/md')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_MD_data(ref_data, tst_data)

    def test_dipole_derivative_cas(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/dipole_deriv_cas')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_MD_data(ref_data, tst_data)

    def test_dipole_derivative_cis(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/dipole_deriv_cis')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_MD_data(ref_data, tst_data)

    def test_geom_import(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/geom_import')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out', coords_file='geom_custom.xyz')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_single_job(ref_data, tst_data)

    def test_cis_import(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/cis')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_single_job(ref_data, tst_data)


if __name__ == '__main__':
    unittest.main()