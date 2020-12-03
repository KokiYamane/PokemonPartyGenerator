import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from operator import itemgetter
from pandas.core.frame import DataFrame


class GeneticAlgorithm():
    def __init__(self, compatibility_table: DataFrame,
                 n_chromosome=200) -> None:
        self.compatibility_table = compatibility_table
        self.n_data = self.compatibility_table.shape[0]
        self.n_chromosome = n_chromosome
        self.chromosomes = [{
            'gene': np.sort(np.random.choice(range(self.n_data), size=6, replace=False)),
            'eval': None} for i in range(n_chromosome)]
        self.len_chromosome = 6

    # 評価関数
    def _eval_chromosome(self, chromosome):
        compatibility_table_part = self.compatibility_table.iloc[chromosome['gene'], :]
        chromosome['eval'] = compatibility_table_part.mean().mean() + 1024
        # if chromosome['eval'] < 0:
        #     chromosome['eval'] = 0
        return chromosome

    # エリート選抜
    def _elite_selection(self, n_selection: int):
        # 染色体のソート
        self.chromosomes = sorted(
            self.chromosomes,
            key=itemgetter('eval'),
            reverse=True)

        # 最優秀染色体の保存
        self.best_chromosome = copy.deepcopy(self.chromosomes[0])

        # エリート選抜
        self.chromosomes = self.chromosomes[:n_selection] + [copy.deepcopy(
            self.best_chromosome) for i in range(self.n_chromosome - n_selection)]

    # ルーレット選択
    def _select_chromosomes_roulette(self):
        sum_dist = sum([x['eval'] for x in self.chromosomes])
        p = [x['eval'] / sum_dist for x in self.chromosomes]
        return np.random.choice(self.n_chromosome, p=p, size=2)

    # 交叉
    def _cross_gene(self, i_ch1, i_ch2):
        candidate = np.union1d(
            self.chromosomes[i_ch1]['gene'],
            self.chromosomes[i_ch2]['gene'])
        gene1 = np.random.choice(
            candidate,
            size=self.len_chromosome,
            replace=False)
        gene2 = np.random.choice(
            candidate,
            size=self.len_chromosome,
            replace=False)
        self.chromosomes[i_ch1]['gene'] = np.sort(gene1)
        self.chromosomes[i_ch2]['gene'] = np.sort(gene2)

    # 突然変異
    def _mutate_chromosome(self, chromosome, p_mutation):
        if np.random.rand() >= p_mutation:
            return chromosome
        index = np.random.choice(self.len_chromosome)
        candidate = list(set(range(self.n_data)) - set(chromosome['gene']))
        new_value = np.random.choice(candidate)
        chromosome['gene'][index] = new_value
        chromosome['gene'] = np.sort(chromosome['gene'])
        return chromosome

    def _chromosome2str(self, chromosome):
        return list(self.compatibility_table.iloc[chromosome['gene'], :].index)

    # 世代ループ
    def calc(self, n_generation=100, p_mutation=0.03,
             p_cross=0.5, p_selection=0.5):
        n_cross = int(self.n_chromosome * p_cross)
        n_selection = int(self.n_chromosome * p_selection)
        self.acc_log = []

        for i_generation in range(n_generation):
            # 染色体の評価
            self.chromosomes = [self._eval_chromosome(
                chromosome) for chromosome in self.chromosomes]

            # エリート選抜
            self._elite_selection(n_selection)

            # 交叉
            for i_cross in range(n_cross):
                i_ch1, i_ch2 = self._select_chromosomes_roulette()
                self._cross_gene(i_ch1, i_ch2)

            # 突然変異
            self.chromosomes = [self._mutate_chromosome(
                chromosome, p_mutation) for chromosome in self.chromosomes]

            # 進捗表示
            print('loop = {:3d}, best_dist = {}, {}'.format
                  (i_generation, self.best_chromosome['eval'],
                   self._chromosome2str(self.best_chromosome)))

            # 全染色体表示
            # for chromosome in self.chromosomes:
            #     print(chromosome['gene'])

            # 評価値の保存
            self.acc_log.append(self.best_chromosome['eval'])

    def plot(self):
        plt.close()
        plt.plot(self.acc_log)
        plt.xlabel('generation')
        plt.ylabel('eval')
        plt.grid()
        plt.savefig('graph.png')
        plt.show()


if __name__ == "__main__":
    # compatibility_table = pd.read_excel(
    #     '対面相性表_16x16_20200213.xlsx', index_col=0, header=0, sheet_name=0)
    # compatibility_table = pd.read_excel(
    #     'pokemon_FFtable.xlsx', index_col=0, header=0, sheet_name=0)
    compatibility_table = pd.read_csv(
        '対面相性表_20x20_20201129.csv', index_col=0, header=0)
    geneticAlgorithm = GeneticAlgorithm(
        compatibility_table, n_chromosome=20)
    geneticAlgorithm.calc(n_generation=50, p_mutation=0.1,
                          p_cross=0.5, p_selection=0.5)
    geneticAlgorithm.plot()
