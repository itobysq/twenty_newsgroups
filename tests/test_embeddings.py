import unittest
import sys
sys.path.append('../scripts')
import tobys_word_embeddings as we

class TestEmbeddings(unittest.TestCase):
    def test_get_embeddings(self):
        train, test = we.get_newsgroups_data()
        self.assertTrue(len(train.data) > 300)
if __name__ == "__main__":
    unittest.main()
