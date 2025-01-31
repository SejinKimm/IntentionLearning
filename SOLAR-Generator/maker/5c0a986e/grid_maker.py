from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random


class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []
        num = 0

        num_samples = kwargs['num_samples']
        max_h, max_w = kwargs['max_grid_dim']
        num_examples = kwargs['num_examples']

        # randomly generate inputs
        while num < num_samples:
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []

            operations = []
            selections = []

            # 공통

            # 색상 2개 선택
            #p_colors = random.sample(range(1, 10), 2)
            p_colors = [1, 2] # 250116_sjkim: fix color

            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                # 내 구현
                # h = np.random.randint(7, max_h)
                h = 10 # 250116_sjkim: fix size
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 두 점이 대각선으로 만나는 지 확인하는 함수
                def is_diagonal_intersect(point1, point2):
                    x1, y1 = point1
                    x2, y2 = point2
                    # 양쪽 대각선 조건을 모두 확인
                    return abs(x1 - x2) == abs(y1 - y2) or x1 + y1 == x2 + y2

                # 랜덤으로 2개 찍기
                points = []
                x1 = np.random.randint(1, h-2)
                y1 = np.random.randint(1, w-2)
                x2, y2 = x1, y1

                while is_diagonal_intersect((x1, y1), (x2, y2)) or is_diagonal_intersect((x1+1, y1), (x2, y2)) or is_diagonal_intersect((x1, y1+1), (x2, y2)):
                    x2 = np.random.randint(1, h-2)
                    y2 = np.random.randint(1, w-2)

                # 랜덤 사각형
                for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    rand_grid[x1+dx][y1+dy] = p_colors[0]
                    rand_grid[x2+dx][y2+dy] = p_colors[1]

                # answer
                # 첫번째 색상은 왼쪽위 두번째 색상을 오른쪽 아래
                answer_grid = rand_grid.copy()

                f_len = min(x1, y1)
                s1 = (x1, y1)
                for i in range(f_len):
                    s1 = (s1[0]-1, s1[1]-1)
                    x, y = s1
                    answer_grid[x][y] = p_colors[0]
                    if (j == num_examples):
                        selections.append([x, y, 0, 0])
                        operations.append(p_colors[0])

                s_len = min(h-x2-1, w-y2-1)
                s2 = (x2+1, y2+1)
                for i in range(s_len-1):
                    s2 = (s2[0]+1, s2[1]+1)
                    x, y = s2
                    answer_grid[x][y] = p_colors[1]
                    if (j == num_examples):
                        selections.append([x, y, 0, 0])
                        operations.append(p_colors[1])

                # ARCLE
                if (j == num_examples):

                    operations.append(34)   # Submit
                    selections.append([0, 0, h-1, w-1])

                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1

                # Example case 저장
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1

            desc = {'id': '5c0a986e',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
