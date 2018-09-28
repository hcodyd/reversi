import sys
import socket
import time
import numpy as np
import math
from random import randint
from copy import deepcopy


class AiGuy:
    rnd: int
    GAME_OVER = -999  # the turn is set to -999 if the game is over
    UPPER_LEFT = [0, 0]
    UPPER_RIGHT = [0, 7]
    LOWER_LEFT = [7, 0]
    LOWER_RIGHT = [7, 7]
    ROWS = 8
    COLS = 8
    me = 0  # this will be set by the server to either 1 or 2
    not_me = 0  # used for keeping track when flipping
    strategicPoints = [
        [100,  -10, 100,  2,  2, 100, -10,  100],
        [-10, -20, -4,  5,  5, -4, -20, -10],
        [100,   -1, 10,  0,  0, 100,  -1,  100],
        [2,     5,  0,  1,  1,  0,   5,   2],
        [2,     5,  0,  1,  1,  0,   5,   2],
        [100,   -1, 100,  0,  0, 100,  -1,  10],
        [-10, -20, -2,  5,  5, -4, -20, -10],
        [100,  -10, 100,  2,  2, 10, -10,  100]
    ]

    t1 = 0.0  # the amount of time remaining to player 1
    t2 = 0.0  # the amount of time remaining to player 2

    state = np.zeros((8, 8))  # this will keep the real state of the game

    def __init__(self):
        # print('Argument List:', str(sys.argv))
        self.play_game(int(sys.argv[2]), sys.argv[1])

    def alpha_beta(self, valid_moves):
        """
        Performs an alpha beta search
        :param valid_moves: the valid moves for current state
        :return: the move that should be taken
        """
        best_move = []
        self.move_search_time = 99999
        if self.rnd < 10:
            depth_to_go = 2  # adjust
        else:
            depth_to_go = 6

        alpha = -math.inf  # starting values
        beta = math.inf
        self.start_time = time.time()
        for i in range(len(valid_moves)):
            best_move.append(self.ab_max(valid_moves[i], depth_to_go, np.copy(self.state), deepcopy(self.rnd)+1, alpha, beta))
        print(time.time()-self.start_time)
        return best_move.index(max(best_move))

    def maximize_ab(self, valid_move, depth_to_go, fake_state, rnd, alpha, beta):
        if depth_to_go == 0 or (time.time() - self.start_time) > self.move_search_time:  # reached maximum depth
            return self.take_random_sampling(fake_state, rnd, self.me)  # when using max, looking from "my" perspective
        new_fake_state = self.take_move(valid_move[0], valid_move[1], self.me, fake_state)
        valid_moves_min = self.get_hypo_valid_moves(rnd + 1, self.not_me, new_fake_state)

        move_returned_values = []
        for i in range(len(valid_moves_min)):
            value = self.minimize_ab(valid_moves_min[i], depth_to_go - 1, np.copy(new_fake_state), rnd + 1, alpha,
                                     beta)
            if value >= beta:
                return value
            alpha = max(value, alpha)
            move_returned_values.append(value)
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amax(move_returned_values)

    def minimize_ab(self, valid_move, depth_to_go, fake_state, rnd, alpha, beta):
        if depth_to_go == 0 or (time.time() - self.start_time) > self.move_search_time:
            return self.take_random_sampling(fake_state, rnd, self.not_me)
        new_fake_state = self.take_move(valid_move[0], valid_move[1], self.not_me, fake_state)
        valid_moves_max = self.get_hypo_valid_moves(rnd + 1, self.me, new_fake_state)

        move_returned_values = []
        for i in range(len(valid_moves_max)):
            value = (
                self.maximize_ab(valid_moves_max[i], depth_to_go - 1, np.copy(new_fake_state), rnd + 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
            move_returned_values.append(value)
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amin(move_returned_values)
    #
    # def min_max(self, valid_moves):
    #     best_move = []
    #     depth_to_go = 4
    #     for i in range(len(valid_moves)):
    #         best_move.append(self.maximize(valid_moves[i], depth_to_go, deepcopy(self.state), self.rnd))
    #     return best_move.index(max(best_move))
    #
    # def maximize(self, valid_move, depth_to_go, fake_state, rnd):
    #     if depth_to_go == 0:
    #         return self.use_heuristic(fake_state, rnd, self.me)
    #     new_fake_state = self.take_move(valid_move[0], valid_move[1], self.me, fake_state)
    #     valid_moves_min = self.get_hypo_valid_moves(rnd + 1, self.not_me, new_fake_state)
    #
    #     move_returned_values = []
    #     for i in range(len(valid_moves_min)):
    #         move_returned_values.append(
    #             self.minimize(valid_moves_min[i], depth_to_go - 1, deepcopy(new_fake_state), rnd + 1))
    #     if len(move_returned_values) == 0:
    #         move_returned_values.append(0)
    #     return np.amax(move_returned_values)
    #
    # def minimize(self, valid_move, depth_to_go, fake_state, rnd):
    #     if depth_to_go == 0:
    #         return self.use_heuristic(fake_state, rnd, self.not_me)
    #     new_fake_state = self.take_move(valid_move[0], valid_move[1], self.not_me, fake_state)
    #     valid_moves_max = self.get_hypo_valid_moves(rnd + 1, self.me, new_fake_state)
    #
    #     move_returned_values = []
    #     for i in range(len(valid_moves_max)):
    #         move_returned_values.append(
    #             self.maximize(valid_moves_max[i], (depth_to_go - 1), deepcopy(new_fake_state), rnd + 1))
    #     if len(move_returned_values) == 0:
    #         move_returned_values.append(0)
    #     return np.amin(move_returned_values)

    def ab_max(self, valid_move, depth_left, board, rnd, alpha, beta):

        # print("----------Max---------- {}".format(depth_left))
        value = -math.inf

        # get new board and new moves for next level down
        new_board = self.take_move(valid_move[0], valid_move[1], self.me, board)

        if depth_left == 0 or (time.time() - self.start_time) > self.move_search_time:  # if at leaf node, return expected utility
            return self.use_heuristic(new_board, rnd, self.me)
        new_moves = self.get_hypo_valid_moves(rnd, self.not_me, new_board)

        # look through all the moves and take the max of of the min
        for x in new_moves:
            value = max(value, self.ab_min(x, depth_left-1, np.copy(new_board), rnd+1, alpha, beta))
            if value >= beta:
                # print("Beta cut off")
                return value  # this is a beta cut off
            alpha = max(alpha, value)
        return value

    def ab_min(self, valid_move, depth_left, board, rnd, alpha, beta):
        # print("--------MIN------------ {}".format(depth_left))
        value = math.inf

        new_board = self.take_move(valid_move[0], valid_move[1], self.not_me, board)

        if depth_left == 0 or (time.time() - self.start_time) > self.move_search_time:  # if at leaf node, return expected utility
            return self.use_heuristic(new_board, rnd, self.not_me)
        new_moves = self.get_hypo_valid_moves(rnd, self.me, new_board)

        for x in new_moves:
            value = min(value, self.ab_max(x, depth_left-1, np.copy(new_board), rnd+1, alpha, beta))
            if value <= alpha:
                # print("Alpha cut off")
                return value  # this is an alpha cut off
            beta = min(beta, value)
        return value

    # ------------------------------------------------------------Heuristics-------------------------------------------------------------

    def take_random_sampling(self, board, rnd, me):
        # print("--------------------------")
        total = 0
        for i in range(0, 10):  # adjust
            utility = self.random_sample(np.copy(board), me, rnd, 2)  # adjust
            # print(utility)
            total += utility
        return total//20

    def random_sample(self, board, me, rnd, depth_left):
        if depth_left == 0:
            return self.use_heuristic(board, rnd, me)

        # print("rnd {} player {}".format(rnd, me))
        # print(board)
        moves = self.get_hypo_valid_moves(rnd, me, board)
        # print("Possible moves {}".format(moves))
        if (len(moves) == 0) or (len(moves) == 1):
            return self.use_heuristic(board, rnd, me)
        else:
            rand = randint(0, len(moves)-1)
        random_move = moves[rand]
        new_board = self.take_move(random_move[0], random_move[1], me, board)
        # print(new_board)
        if me == 1:
            me = 2
        else:
            me = 1
        return self.random_sample(np.copy(new_board), me, rnd+1, depth_left-1)

    def use_heuristic(self, board, rnd, me):
        """
        Returns an expected utility of a given board from the perspective of player "me".
        :param board: the hypothetical board state
        :param rnd: the hypothetical round
        :param me: 1 or 2 depending on player perspective
        :return: a utility (int)
        """
        opp = 1 if me == 2 else 2

        if rnd < 44:
            stability = self.stability(board, me)
            strategic = self.strategic_points(board, me)
            mobility = self.mobility(board, rnd, opp)
            flipped_with_move = self.score_board(board, me)
            # print(strategic * .8 + stability * .5 + mobility * .8 + flipped_with_move * .2)
            return strategic * .8 + stability * .5 + mobility * .8 + flipped_with_move * .5
        else:
            points = self.score_board(board, me) * (self.stability(board, me) + 1) * self.strategic_points(board, me)
            return points

    def strategic_points(self, board, me):
        strategic_points = 0
        for i in range(len(board[0])):
            for j in range(len(board[0])):
                if board[1, j] == me:
                    strategic_points += self.strategicPoints[i][j]
        return strategic_points

    @staticmethod
    def score_board(board, me):
        board = np.matrix(board)
        player_one_score = np.count_nonzero(board == 1)
        player_two_score = np.count_nonzero(board == 2)
        if me == 1:
            return player_one_score - player_two_score
        else:
            return player_two_score - player_one_score

    def mobility(self, board, rnd, me):
        """
        Returns the number of available moves to the given player "me" in the hypothetical board.
        :param board: the hypothetical board
        :param rnd: the hypothetical round it is
        :param me: the player to test for
        :return: the number of available moves (int)
        """
        oppMove = 1 if len(self.get_hypo_valid_moves(rnd, self.not_me, board)) == 0 else len(self.get_hypo_valid_moves(rnd, self.not_me, board))
        return len(self.get_hypo_valid_moves(rnd, self.me, board))/oppMove

    def stability(self, board, me):
        """
        Goes through all "my" pieces and checks if they are stable.
        :param board: the current board
        :param me: either player 1 or 2
        :return: the number of stable stones
        """
        stable_disks = 0
        for i in range(8):
            for j in range(8):
                if self.is_stable_disk(i, j, me, board):
                    stable_disks += 1
        return stable_disks

    # -----------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def is_stable_dir(row, col, dir_x, dir_y, board):
        """
        Returns the board if the move at given row and col is taken
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param dir_x: row direction to look
        :param dir_y: col direction to look
        :param board: if changing a hypothetical state
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + dir_y * i
            c = col + dir_x * i

            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            sequence.append(board[r, c])

        empty = 0
        for val in sequence:
            if val == 0:
                empty += 1

        if empty == 1:  # if there is only one un-taken spot, that is the one open spot for this move
            return True
        else:
            return False

    @staticmethod
    def is_dir_me(row, col, dir_x, dir_y, board, me):
        """
        Returns True if all the pieces in that direction are filled by "me"
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param dir_x: row direction to look
        :param dir_y: col direction to look
        :param board: if changing a hypothetical state
        :param me: 1 or 2
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + dir_y * i
            c = col + dir_x * i

            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            sequence.append(board[r, c])

        if 0 in sequence:
            return False
        elif (me == 1) and (2 in sequence):
            return False
        elif (me == 2) and (1 in sequence):
            return False
        else:
            return True

    def is_stable_disk(self, x, y, me, board):
        """
        Check if the given disk is stable (can't be flipped).
        :param x: the x coord of the disk
        :param y: the y coord of the disk
        :param me: the player (1 or 2)
        :param board: the board
        :return: true or false
        """
        disk = [x, y]

        # all corners are stable
        if disk == self.UPPER_RIGHT:
            return True
        if disk == self.UPPER_LEFT:
            return True
        if disk == self.LOWER_RIGHT:
            return True
        if disk == self.LOWER_LEFT:
            return True

        # first check if any corner is taken by "me"
        # then check if the square of stones back to the corner is all "my" color
        # if so, I am in an anchored corner
        if (board[0, 0] == me) and (
                self.is_dir_me(x, y, -1, 0, board, me) and (self.is_dir_me(x, y, 0, -1, board, me))):
            return True
        if (board[0, 7] == me) and (self.is_dir_me(x, y, -1, 0, board, me) and (self.is_dir_me(x, y, 0, 1, board, me))):
            return True
        if (board[7, 0] == me) and (self.is_dir_me(x, y, 1, 0, board, me) and (self.is_dir_me(x, y, 0, -1, board, me))):
            return True
        if (board[7, 7] == me) and (self.is_dir_me(x, y, 1, 0, board, me) and (self.is_dir_me(x, y, 0, 1, board, me))):
            return True

        # check if am on a stable side
        if x == 0 and self.is_stable_dir(0, 0, 0, 1, board):
            return True
        if x == 7 and self.is_stable_dir(7, 0, 0, 1, board):
            return True
        if y == 0 and self.is_stable_dir(0, 0, 1, 0, board):
            return True
        if y == 7 and self.is_stable_dir(0, 7, 1, 0, board):
            return True

        # checks if every direction around it is filled (DO LAST)
        for dir_x in range(-1, 2):  # check in all directions
            for dir_y in range(-1, 2):
                if (dir_x == 0) and (dir_y == 0):  # no need to check curr place
                    if not self.is_stable_dir(x, y, dir_x, dir_y, board):
                        return False
        return True

    @staticmethod
    def flip_in_dir(row, col, dir_x, dir_y, me, board):
        """
        Returns the board if the move at given row and col is taken
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param dir_x: row direction to look
        :param dir_y: col direction to look
        :param me: the given player
        :param board: if changing a hypothetical state
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + dir_y * i
            c = col + dir_x * i

            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            sequence.append(board[r, c])

        stones_to_flip = 0
        can_flip = False
        for i in range(len(sequence)):
            if me == 1:
                if sequence[i] == 1:
                    can_flip = True
                    break
                if sequence[i] == 2:
                    stones_to_flip += 1
                if sequence[i] == 0:
                    stones_to_flip = 0
                    break
            else:  # p2
                if sequence[i] == 2:
                    can_flip = True
                    break
                if sequence[i] == 1:
                    stones_to_flip += 1
                if sequence[i] == 0:
                    stones_to_flip = 0
                    break
        if can_flip:
            for i in range(1, stones_to_flip + 1):
                board[row + dir_y * i, col + dir_x * i] = me

    def take_move(self, row, col, me, board):
        """
        Checks if an unoccupied square can be played by the given player
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param me: the player
        :param board: the state to change
        :return: True or False
        """
        for dir_x in range(-1, 2):  # check in all directions
            for dir_y in range(-1, 2):
                if (dir_x == 0) and (dir_y == 0):  # no need to check curr place
                    board[row, col] = me
                else:
                    self.flip_in_dir(row, col, dir_x, dir_y, me, board)
        return board

    def get_hypo_valid_moves(self, rnd, me, board):
        """
        Returns the valid moves for the hypothetical board state.
        :param rnd: the round (hypothetical)
        :param me: the player taking next move
        :param board: the hypothetical board
        :return: the valid moves
        """
        valid_moves = []
        if rnd < 4:
            if board[3, 3] == 0:
                valid_moves.append([3, 3])
            if board[3, 4] == 0:
                valid_moves.append([3, 4])
            if board[4, 3] == 0:
                valid_moves.append([4, 3])
            if board[4, 4] == 0:
                valid_moves.append([4, 4])
        else:
            for i in range(8):
                for j in range(8):
                    if board[i, j] == 0:
                        if self.hypo_could_be(i, j, me, board):
                            valid_moves.append([i, j])

        return np.array(valid_moves)

    def hypo_could_be(self, row, col, me, board):
        """
        Checks if an unoccupied square can be played by the given player
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param me: the player (1 or 2)
        :param board: the hypothetical board
        :return: True or False
        """
        for dir_x in range(-1, 2):
            for dir_y in range(-1, 2):
                if (dir_x == 0) and (dir_y == 0):  # no need to check curr place
                    continue

                if self.hypo_check_dir(row, col, dir_x, dir_y, me, board):
                    return True

        return False

    @staticmethod
    def hypo_check_dir(row, col, dir_x, dir_y, me, board):
        """
        Figures out if the move at row, col is valid for the given player in the given direction.
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param dir_x: the x direction
        :param dir_y: the y direction
        :param me: the given player
        :param board: the hypothetical board
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + dir_y * i
            c = col + dir_x * i
            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            sequence.append(board[r, c])

        count = 0
        for i in range(len(sequence)):
            if me == 1:
                if sequence[i] == 2:
                    count = count + 1
                else:
                    if (sequence[i] == 1) and (count > 0):
                        return True
                    break
            else:
                if sequence[i] == 1:
                    count = count + 1
                else:
                    if (sequence[i] == 2) and (count > 0):
                        return True
                    break

        return False

    # ------------------------------------- Code that affects the state -----------------------------------------------

    def check_dir(self, row, col, dir_x, dir_y, me):
        """
        Figures out if the move at row, col is valid for the given player in the given direction.
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param dir_x: the x direction
        :param dir_y: the y direction
        :param me: the given player
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + dir_y * i
            c = col + dir_x * i
            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            sequence.append(self.state[r, c])

        count = 0
        for i in range(len(sequence)):
            if me == 1:
                if sequence[i] == 2:
                    count = count + 1
                else:
                    if (sequence[i] == 1) and (count > 0):
                        return True
                    break
            else:
                if sequence[i] == 1:
                    count = count + 1
                else:
                    if (sequence[i] == 2) and (count > 0):
                        return True
                    break

        return False

    def could_be(self, row, col, me):
        """
        Checks if an unoccupied square can be played by the given player
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param me: the player
        :return: True or False
        """
        for dir_x in range(-1, 2):
            for dir_y in range(-1, 2):
                if (dir_x == 0) and (dir_y == 0):  # no need to check curr place
                    continue

                if self.check_dir(row, col, dir_x, dir_y, me):
                    return True

        return False

    def get_valid_moves(self, rnd, me):
        """
        Generates the set of valid moves for the player
        :param rnd: what number round the game is currently at.
        :param me: the player to check the moves for (1 or 2)
        :return: A list of valid moves
        """
        valid_moves = []
        if rnd < 4:
            if self.state[3, 3] == 0:
                valid_moves.append([3, 3])
            if self.state[3, 4] == 0:
                valid_moves.append([3, 4])
            if self.state[4, 3] == 0:
                valid_moves.append([4, 3])
            if self.state[4, 4] == 0:
                valid_moves.append([4, 4])
        else:
            for i in range(8):
                for j in range(8):
                    if self.state[i, j] == 0:
                        if self.could_be(i, j, me):
                            valid_moves.append([i, j])

        return valid_moves

    @staticmethod
    def init_client(me, host):
        """
        Establishes a connection with the server.
        :param me: ?
        :param host: ?
        :return: a "sock"
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_address = (host, 3333 + me)
        print('starting up on %s port %s' % server_address, file=sys.stderr)
        sock.connect(server_address)
        info = sock.recv(1024)
        print("Sock info: {}".format(info))

        return sock

    def read_message(self, sock):
        """
        Reads messages from the server that tell the player whose turn it is and what moves are made.
        :param sock: ?
        :return: the current turn and the round
        """
        message = sock.recv(1024).decode("utf-8").split("\n")

        turn = int(message[0])

        if turn == self.GAME_OVER:
            time.sleep(1)
            sys.exit()

        rnd = int(message[1])

        # update the global state variable
        count = 4
        for i in range(8):
            for j in range(8):
                self.state[i, j] = int(message[count])
                count += 1
        self.rnd = rnd  # update the global round variable
        return turn, rnd

    def play_game(self, me, host):
        """
        Establishes a connection with the server.
        Then plays whenever it is this player's turn.
        :param me: the player number of the AI
        :param host: ?
        :return: None
        """
        sock = self.init_client(me, host)  # Establish connection
        self.me = me
        if me == 1:
            self.not_me = 2
        else:
            self.not_me = 1

        while True:
            status = self.read_message(sock)  # get update from server
            if status[0] == me:  # status[0] has turn
                # print("-----------------Round {}-----------------".format(status[1]))  # status[1] has round
                # print("My turn.")
                # self.print_game_state()

                valid_moves = self.get_valid_moves(status[1], me)  # status[1] has round
                my_move = self.alpha_beta(valid_moves)

                # print("Valid moves: {}".format(valid_moves))
                # print("Selected move: {}".format(valid_moves[my_move]))

                # Send selection
                selection = str(valid_moves[my_move][0]) + "\n" + str(valid_moves[my_move][1]) + "\n"
                sock.send(selection.encode("utf-8"))
                # print("------------------------------------------")
            else:
                pass
                # print("Cody player's turn.")  # Always prints twice?


if __name__ == "__main__":
    """
    Call from command line like this: python AIGuy.py [ip_address] [player_number]
    ip_address is the ip_address on the computer the server was launched on. Enter "localhost" if on same computer
    player_number is 1 (for the black player) and 2 (for the white player)
    """
    # print('Argument List:', str(sys.argv))
    # play_game(int(sys.argv[2]), sys.argv[1])
    AiGuy()
