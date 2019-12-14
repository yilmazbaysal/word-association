import csv

from scipy.sparse import load_npz


class DataPreparer:

    #
    def __init__(self):
        self.vocabulary = {}
        self.word_pairs = []

    #
    @staticmethod
    def read_matrix_file(file_path: str):
        return load_npz(file_path).toarray()

    #
    def __read_vocabulary_file(self, file_path: str):
        with open(file_path, 'r') as f:
            for index, line in enumerate(f.read().split()):
                token, count = line.split(';')
                self.vocabulary[token] = (index, int(count))

    #
    def read_word_pairs_file(self, file_path: str):
        self.__read_vocabulary_file('../dataset/vocabular.list')

        sum_all_human_judgements = 0
        with open(file_path, 'r') as f:
            for row in csv.reader(f):
                index_1, count_1 = self.vocabulary[row[0]]
                index_2, count_2 = self.vocabulary[row[1]]

                # Sum up the human judgements
                sum_human_judgements = 0
                for i in range(2, len(row)):
                    sum_human_judgements += int(row[i])

                avg_human_judgements = sum_human_judgements / (len(row) - 2)
                self.word_pairs.append((index_1, index_2, avg_human_judgements))
                sum_all_human_judgements += avg_human_judgements

        return self.word_pairs, sum_all_human_judgements / len(self.word_pairs)

