import random
import my_game
from constants import BLACK_EDGE as black_edge
from constants import WHITE_EDGE as white_edge

class SerialNumber:
    """
        实现每道小题目的逻辑
    """
    def __init__(self) -> None:
        # 初始化ai对弈模块
        self.mygame = my_game.PlayGame()
        
    def one(self, board_centre_list: list[list], black_chess_list: list) -> dict:
        """
            装置能将任意 1颗黑棋子放置到 5 号方格中
            :param board_centre_list: 棋盘中心坐标列表
            :param black_chess_list: 黑色棋子坐标列表
            :return: 机械运动的 起始与终止 坐标
        """
        # 得到处于放置区的棋子坐标列表
        black_chess_list = self.remove_excess_chess(black_chess_list, "black")
        
        # 打乱棋子顺序，随机获得
        random.shuffle(black_chess_list)

        data_dir = {
            "from": black_chess_list[0],
            "to": board_centre_list[1][1]
        }

        return data_dir
    
    def two_and_three(self, cell: int, board_centre_list: list[list], chess_list: list) -> dict:
        """
            装置能将任意 2 颗黑棋子和 2 颗白棋子依次放置到指定方格中
            :param cell: 指定放入的格子
            :param board_centre_list: 棋盘中心坐标列表
            :param chess_list: 本次移动的棋子坐标列表
            :return: 机械运动的 起始与终止 坐标
        """
        if chess_list[0][1] < 115:
            chess_list = self.remove_excess_chess(chess_list, "white")
        else:
            chess_list = self.remove_excess_chess(chess_list, "black")

        # 打乱棋子顺序，随机获得
        random.shuffle(chess_list)
        cell -= 1

        row = cell // 3
        col = cell % 3

        data_dir = {
            "from": chess_list[0],
            "to": board_centre_list[row][col]
        }

        return data_dir

    def four(self, board_centre_list: list[list], chess_list: list, board: list) -> dict:
        """
            装置执黑棋先行与人对(第 1 步方格可设置)，若人应对的第 1 步白棋有错误，装置能获胜
            :param board_centre_list: 棋盘中心坐标列表
            :param black_chess_list: 黑色棋子坐标列表
            :param board: 当前的下棋棋盘
            :return: 机械运动的 起始与终止 坐标
        """
        # 得到处于放置区的棋子坐标列表
        chess_list = self.remove_excess_chess(chess_list, "black")
        
        data_dir = {}
        # 人机对弈得到行列信息
        data_dir = self.mygame.play_game(board)

        # 将行列信息转换为坐标信息
        if ("from" in data_dir) and ("to" in data_dir):
            data_dir = self.convert(board_centre_list, chess_list, data_dir)
        
        return data_dir

    def five(self, board_centre_list: list[list], chess_list: list, board: list) -> dict:
        """
            人执黑棋先行，装置能正确放置白棋子以保持不输棋
            :param board_centre_list: 棋盘中心坐标列表
            :param black_chess_list: 黑色棋子坐标列表
            :param board: 当前的下棋棋盘
            :return: 机械运动的 起始与终止 坐标
        """
        # 得到处于放置区的棋子坐标列表
        chess_list = self.remove_excess_chess(chess_list, "white")
        data_dir = {}
        # 人机对弈得到行列信息
        data_dir = self.mygame.play_game(board)


        # 将行列信息转换为坐标信息
        if ("from" in data_dir) and ("to" in data_dir):
            data_dir = self.convert(board_centre_list, chess_list, data_dir)
        
        return data_dir
    def six(self):
        pass

    def remove_excess_chess(self, chess_list, color) -> list:
        """
            去除多余的棋子坐标信息，留下放置处的棋子坐标
        """
        if color == "black":
            points = [point for point in chess_list if point[0]<black_edge]
        else:
            points = [point for point in chess_list if point[0]>white_edge]

        return points
    

    def convert(self, board_centre_list: list[list], my_chess_site: list, data_dir: dict) -> dict:
        """
            将字典中的行列数据转换为坐标信息
        """
        # 提取字典中的row和col
        if data_dir['from'] == (-1, -1):
            # 得到可以使用的棋子坐标列表
            random.shuffle(my_chess_site)
            data_dir['from'] = my_chess_site[0]
        else:
            # 本次棋盘有误，矫正棋盘信息
            row = data_dir['from'][0]
            col = data_dir['from'][1]
            data_dir['from'] = board_centre_list[row][col]

        row = data_dir['to'][0]
        col = data_dir['to'][1]
        data_dir['to'] = board_centre_list[row][col]

        return data_dir
         