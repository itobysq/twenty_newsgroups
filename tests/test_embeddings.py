import unittest
import sys
sys.path.append('../scripts')
import classifier_helpers as ch

class TestEmbeddings(unittest.TestCase):

    def test_get_embeddings(self):
        train, test = ch.get_newsgroups_data()
        self.assertTrue(len(train.data) > 300)

    def test_average_vectors(self):
        train, test = ch.get_newsgroups_data()
        model = ch.build_model_from_google()
        vectors = ch.average_docs(train, model, vec_length=300)
        self.assertTrue(len(vectors[0] == 300))

    def test_data_padder(self):
        raise NotImplementedError
if __name__ == "__main__":
    unittest.main()
