import unittest
import sys 

sys.path.append('../package/agritech_api_package01')
from get_bounds import Bounds
from file_handler import FileHandler

class TestFileHandler(unittest.TestCase):
    
    def test_set_output_file_name(self):
        file = FileHandler()
        actual = file.set_output_file_name('iowa')
        expected = ('laz/iowa.laz','tif/iowa.tif')
        self.assertEqual(actual, expected)

    def test_set_bounds(self):
        bounds = Bounds()
        actual = bounds.set_bounds(-93.756155, 41.918015, -93.747334, 41.921429)
        expected = (([-10436887.43333523,-10435905.484106943],[5148706.389047224,5149217.145836504]))
        self.assertEqual(actual, expected)
        

        
if __name__ == '__main__':
  unittest.main()