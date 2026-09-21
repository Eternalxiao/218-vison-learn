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
    _error = []    # 记录错误位置

    def __init__(self):
        pass

    # 打印棋盘
    def print_board(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for row in self._board:
            print('|'.join([symbols[cell] for cell in row]))
            print('-' * 5)

    # 更新棋盘
    def place_move(self, row, col, player):
        if self._board[row][col] == 0:
            self._board[row][col] = player
            return True
        return False  # 位置已被占用

    # 判断游戏是否结束
    def check_win(self, player):
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
    def check_draw(self):
        for row in self._board:
            if 0 in row:
                return False
        return True

    # ai决策下一步位置
    def ai_move(self):
        # 1. 尝试直接获胜
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0:
                    self._board[i][j] = 2  # AI是2
                    if self.check_win(2):  # ai下棋后判断是否胜利
                        return (i, j)   # 胜利，返回位置
                    self._board[i][j] = 0  # 恢复

        # 2. 阻止玩家获胜
        for i in range(3):
            for j in range(3):
                if self._board[i][j] == 0:
                    self._board[i][j] = 1  # 玩家是1
                    if self.check_win(1):  # 玩家下棋后判断是否胜利
                        self._board[i][j] = 2  # ai堵住位置
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

    # 开始游戏
    def play_game(self):
        players = ['玩家', 'AI']
        current_player = 0  # 0:玩家，1:AI

        while True:
            self.print_board()
            if current_player == 0:
                # 玩家回合
                row = int(input("输入行 (0-2): "))
                col = int(input("输入列 (0-2): "))
                if self.place_move(row, col, 1):
                    current_player = 1
                else:
                    print("位置无效！")
            else:
                # AI回合
                print("AI正在思考...")
                # if self.check_error():  # 错误位置检查
                #     # 修正错误位置
                #     self.correct_mistake()
                #     current_player = 0
                #     break
                move = self.ai_move()  # 或 ai_move_minimax(board)
                print(f"ai下棋到：{move=}")
                self.place_move(move[0], move[1], 2)
                current_player = 0

            # 检查胜负
            if self.check_win(1):
                self.print_board()
                print("玩家获胜！")
                break
            elif self.check_win(2):
                self.print_board()
                print("AI获胜！")
                break
            elif self.check_draw():
                self.print_board()
                print("平局！")
                break
            # 更新上一局棋盘
            # self._last_board = [row[:] for row in self._board]

    # 检查错误位置
    def check_error(self):
        for i in range(3):
            for j in range(3):
                if self._board[i][j] != self._last_board[i][j]:
                    self._error.append((i,j))   # 记录不同的位置

        if len(self._error) != 1:  # 错误不是一个，说明有错误
            return True

        self._error = []    # 没有或只有一个错误位置，说明没下或者下一个正确的，将错误位置恢复为空
        return False

    # 修正错误位置
    def correct_mistake(self):
        data_dir = {}
        if len(self._error) == 0:  # 对手没下棋，不用修正
            return data_dir

        for pos in self._error:
            self._board[pos[0]][pos[1]] = self._last_board[pos[0]][pos[1]]

        # 对对手下棋后的修正
        # 如果第一错误位置上次空，本次有 --->>> 第一错误为移动的起始端， 第二错误为移动的终止端 ==》》假设只有俩个
        if (self._last_board[self._error[0][0], self._error[0][1]] == 0) \
                and((self._board[self._error[0][0], self._error[0][1]] == 1) or \
                    (self._board[self._error[0][0], self._error[0][1]] == 2)):
            data_dir = {
                'from': self._error[0],
                'to': self._error[1]
            }
        else:
            data_dir = {
                'from': self._error[1],
                'to': self._error[0]
            }

        self._error = []     # 修正完毕，将错误位置恢复为空
        return data_dir

if __name__ == '__main__':
    game = PlayGame()
    # 开始游戏
    game.play_game()