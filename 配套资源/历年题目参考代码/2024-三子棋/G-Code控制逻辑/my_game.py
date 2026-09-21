import random

class PlayGame:
    # 初始化空棋盘
    _board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    _last_board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    _error = []    # 记录差异位置的列表
    _current_player = 0     # 0:玩家，1:AI
    def __init__(self):
        pass

    # 开始游戏
    def play_game(self, new_board, from_point:tuple = (-1, -1)) -> dict:
        """
            实现ai博弈的主程序
            new_board是对弈棋盘数据
            from_point(1-, -1)是默认的拿棋位置，如果本次棋盘不合法则无用

            返回row和col的数据字典：
                data_dir = {
                    'from': (i, j),
                    'to': (i, j)
                }
                from 是起始位置，to 是终止位置
            如果为空表明：对手没有下棋，重新开始游戏
        """

        players = ['玩家', 'AI']
        # 初始化返回的数据字典
        data_dir = {}

    
        # 更新当前棋盘的数据，如果没有传入的棋盘数据，则使用上局的棋盘
        self._board = [row[:] for row in new_board] if new_board else self._board

        # 显示本次将要运算的棋盘
        self.print_board()

        # 由于传入的已经是判断好的棋盘，所以每次调用必定是ai回合
        print("AI正在思考...")
        
        # 检查胜负
        if self.check_win(2):
            self.print_board()
            print("AI获胜！")
            self._last_board = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            return {"game": "I have won"}
        elif self.check_win(1):
            self.print_board()
            print("玩家获胜！")
            self._last_board = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            return {"game": "You have won"}
        
        elif self.check_draw():
            self.print_board()
            print("平局！")
            self._last_board = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            return {"game": "over"}

        # 检查棋盘的合法性，判断对手是否正常下棋
        if self.check_error():  # 差异检查
            # 如果有两个以上的差异
            data_dir.update(self.get_mistake())   # 获取本次差异中需要修正的数据
            print("恢复棋盘棋子位置...")
            return data_dir
        # 计数现在棋子数目，为了正常执行机器执棋先行
        flag = 0
        for row in self._last_board:
            for cul in row:
                if cul != 0:
                    flag = 1
                
        if len(self._error) == 0 and flag == 1:   # 对手还没有下棋
            print("玩家没有下棋，等待下一次命令")
            self._error = []
            return {"game": "you need play chess"}

        # 对手正常下棋，ai思考下棋位子
        move = self.ai_move()       # 获得ai想要下棋的到的棋盘位置：i行j列
        
        # ai成功下棋，更新上次棋盘数据
        if self.place_move(move[0], move[1], 2):
            print(f"ai下棋到：{move=}")
            self._last_board = [row[:] for row in self._board]
        else:
            print(f"ai决策不合理，更新棋盘失败")

        # 得到ai下棋的棋子去向 'to'  
        data_dir = {
            'from': from_point,
            'to': move
        }
        # 将差异列表滞空
        self._error = []
        return data_dir


    # 打印棋盘
    def print_board(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("this")
        for row in self._board:
            print('|'.join([symbols[cell] for cell in row]))
            print('-' * 5)

    # 如果位置合理就更新数据，否则返回 False
    def place_move(self, row, col, player) -> bool:
        if self._board[row][col] == 0:
            self._board[row][col] = player
            return True
        return False  # 位置已被占用

    # 判断游戏是否结束
    def check_win(self, player) -> bool:
        # 检查行
        for row in self._board:
            if row.count(player) == 3:
                return True
        # 检查列
        for col in range(3):
            if self._board[0][col] == player and self._board[1][col] == player and self._board[2][col] == player:
                return True
        # 检查对角线
        if (self._board[0][0] == player and self._board[1][1] == player and self._board[2][2] == player) or \
                (self._board[0][2] == player and self._board[1][1] == player and self._board[2][0] == player):
            return True
        return False

    # 判断游戏是否平局
    def check_draw(self) -> bool:
        for row in self._board:
            if 0 in row:
                return False
        return True

    # ai决策下一步位置
    def ai_move(self) -> tuple:
        # 1. 尝试直接获胜
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0:
                    self._board[i][j] = 2  # AI是2
                    if self.check_win(2):  # ai下棋后判断是否胜利
                        self._board[i][j] = 0   # 确定胜利回滚为0等待放置
                        return (i, j)       # 返回位置
                    self._board[i][j] = 0  # 恢复

        # 2. 阻止玩家获胜
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0:
                    self._board[i][j] = 1  # 玩家是1
                    if self.check_win(1):  # 玩家下棋后判断是否胜利
                        self._board[i][j] = 0  # ai准备堵住到此位置，回滚为0等待放置
                        return (i, j)   # 堵住，返回位置
                    self._board[i][j] = 0

        # 3. 占中心
        if self._board[1][1] == 0:
            return (1, 1)

        # 4. 占角落
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        empty_corners = [pos for pos in corners if self._board[pos[0]][pos[1]] == 0]
        if empty_corners:
            return random.choice(empty_corners)

        # 5. 随机选择剩余位置
        empty_cells = [(i, j) for i in range(3) for j in range(3) if self._board[i][j] == 0]
        return random.choice(empty_cells)

    # 检查俩次棋盘的差异，判断是否需要修正
    def check_error(self) -> bool:
        """
        检查本次棋盘和上次棋盘的差异，将差异位置保存在self._error列表中
        对差异数量进行判断，如果只有一个差异，说明对手正常下棋，返回False
        如果存在多个差异，说明对手下错了，返回True
        :return: True: 存在错误需要修正， False: 不需要修正
        """
        for i in range(3):
            for j in range(3):
                if self._board[i][j] != self._last_board[i][j]:
                    self._error.append((i,j))   # 记录不同的位置

        if len(self._error) > 1:  # 差异大于一个，说明有错误
            return True
        
        # 没有或只有一个差异位置，说明没下或者下一个正确的，不需要修正，将错误位置恢复为空
        return False

    # 获得需要修正棋子的信息
    def get_mistake(self) -> dict:
        """
            获得需要修正棋子的位置，返回数据字典：from 和 to
        """
        print(f"{self._error=}")
        data_dir = {}

        # 错误数据大于一，完成题目的复位要求
        # 如果第一个差异的位置：上次空，本次有 ---说明>>> 第一差异位置为from， 第二错误为移动to   ---》 前提：对手只将棋盘的一个棋子移动一个位置
        if (self._last_board[self._error[0][0]][self._error[0][1]] == 0) and\
                    ((self._board[self._error[0][0]][self._error[0][1]] == 1) or \
                    (self._board[self._error[0][0]][self._error[0][1]] == 2)):
            data_dir = {
                'from': self._error[0],
                'to': self._error[1]
            }
        else:
            data_dir = {
                'from': self._error[1],
                'to': self._error[0]
            }
        
        # 成功得到了需要修正的数据位置，将棋盘恢复到上次
        for pos in self._error:
            self._board[pos[0]][pos[1]] = self._last_board[pos[0]][pos[1]]

        self._error = []     # 修正信息完毕，将差异列表置空
        return data_dir
