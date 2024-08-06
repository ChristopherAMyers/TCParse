import unittest
import os
from tcparse import TCParser
import json

#   turn on to print which keys and frames are being checked
PRINT_ASSERT = False

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
            ref_keys = list(ref_job.keys())
            tst_keys = list(tst_job.keys())
            self.assertListEqual(ref_keys, tst_keys)

            #   check each object in the job
            for key in ref_keys:
                if PRINT_ASSERT: print(" - ASSERTING KEY: ", key)
                self.assertAlmostEqual(ref_job[key], tst_job[key], places=15)

    def _test_single_job(self, ref_job, tst_job):
        
        ref_keys = list(ref_job.keys())
        tst_keys = list(tst_job.keys())
        self.assertListEqual(ref_keys, tst_keys)

        #   check each object in the job
        for key in ref_keys:
            if PRINT_ASSERT: print(" - ASSERTING KEY: ", key)
            self.assertAlmostEqual(ref_job[key], tst_job[key], places=15)

    def test_MD(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/md')
        parser = TCParser()
        tst_data = parser.parse_from_file('tc.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('tc.json') as file:
            ref_data = json.load(file)

        self._test_MD_data(ref_data, tst_data)

    def test_dipole_derivative(self):
        os.chdir(self._orig_dir)
        os.chdir('jobs/dipole_deriv')
        parser = TCParser()
        tst_data = parser.parse_from_file('dipole_deriv.out')
        tst_data = json.loads(json.dumps(tst_data))
        with open('dipole_deriv.json') as file:
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