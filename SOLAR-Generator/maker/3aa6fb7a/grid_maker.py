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
            selections: List[NDArray] = []
            operations: List[NDArray] = []

            # 공통

            # 색상 2개 선택
            p_colors = random.sample(range(1, 10), 2)

            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                # 내 구현
                h = np.random.randint(10, max_h)
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 랜덤으로 n개 찍기
                num_p = random.choice(range(2, h * h // 16)) 

                points = []
                iter_num = 0 
                while iter_num != num_p:
                    x = random.randint(1,h-4)
                    y = random.randint(1,w-4)
                    new_point = (x, y)
                    
                    iter_num += 1
                    
                    if all(abs(new_point[0] - point[0]) >= 4 and abs(new_point[1] - point[1]) >= 4 for point in points):
                        points.append(new_point)
                    else:
                        continue

                # 4가지 경우 (ㄱ,ㄴ,ㄱ 반대, ㄴ반대 )
                for x, y in points:
                    color_code = np.random.randint(4)
                    if color_code == 0: # ㄱ
                        rand_grid[x][y] = p_colors[0]
                        rand_grid[x][y+1] = p_colors[0]
                        rand_grid[x+1][y+1] = p_colors[0]
                    elif color_code == 1: # ㄴ
                        rand_grid[x][y] = p_colors[0]
                        rand_grid[x+1][y] = p_colors[0]
                        rand_grid[x+1][y+1] = p_colors[0]
                    elif color_code == 2: # ㄱ 반대
                        rand_grid[x][y] = p_colors[0]
                        rand_grid[x+1][y] = p_colors[0]
                        rand_grid[x][y+1] = p_colors[0]
                    elif color_code == 3: # ㄴ 반대 
                        rand_grid[x+1][y] = p_colors[0]
                        rand_grid[x][y+1] = p_colors[0]
                        rand_grid[x+1][y+1] = p_colors[0]
                        
                # answer - 3개에 대한 색이 같아야 한다. 문제 넣을 때도 주의하자! 
                answer_grid = rand_grid.copy()
                    
                # 나머지 채우기 
                value_list = []
                random.shuffle(points)
                for x, y in points:   
                    value_list = [answer_grid[x,y],answer_grid[x+1,y],answer_grid[x,y+1],answer_grid[x+1,y+1]]
                    
                    # 사각형에서 다른 값의 인덱스를 찾고 채워 넣기 
                    for i in range(len(value_list)):
                        if value_list.count(value_list[i]) == 3:
                            different_value = value_list[i]
                        else:
                            different_index = i    
                    
                    if different_index == 0:
                        answer_grid[x,y] = p_colors[1]
                        if (j == num_examples):  
                            selections.append([x, y, 0, 0])
                            operations.append(p_colors[1])   # Color
                    elif different_index == 1:
                        answer_grid[x+1,y] = p_colors[1]
                        if (j == num_examples):  
                            selections.append([x+1, y, 0, 0])
                            operations.append(p_colors[1])   # Color
                    elif different_index == 2:
                        answer_grid[x,y+1] = p_colors[1]
                        if (j == num_examples):  
                            selections.append([x, y+1, 0, 0])
                            operations.append(p_colors[1])   # Color
                    elif different_index == 3:
                        answer_grid[x+1,y+1] = p_colors[1]
                        if (j == num_examples):  
                            selections.append([x+1, y+1, 0, 0])
                            operations.append(p_colors[1])   # Color
                            
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

            desc = {'id': '3aa6fb7a',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat