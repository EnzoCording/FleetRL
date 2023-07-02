import datetime
import unittest
import pandas as pd

from FleetRL.fleet_env.fleet_environment import FleetEnv


class MyTestCase(unittest.TestCase):
    # def test_something(self):
    #    self.assertEqual(True, False)  # add assertion here

    def test_soc_mod_on_reset(self):
        env = FleetEnv()
        env.reset()
        env.episode.start_time=pd.to_datetime('2020-01-06 10:30:00')
        self.assertTrue(env.info["soc_mod"])


if __name__ == '__main__':
    unittest.main()
