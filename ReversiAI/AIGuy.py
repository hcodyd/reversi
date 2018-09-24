import sys
import socket
import time
from random import randint
import numpy as np
from copy import deepcopy
import math


class AiGuy:
    GAME_OVER = -999  # the turn is set to -999 if the game is over
    UPPER_LEFT = [0, 0]
    UPPER_RIGHT = [0, 7]
    LOWER_LEFT = [7, 0]
    LOWER_RIGHT = [7, 7]
    me = 0  # this will be set by the server to either 1 or 2
    not_me = 0  # used for keeping track when flipping

    t1 = 0.0  # the amount of time remaining to player 1
    t2 = 0.0  # the amount of time remaining to player 2

    state = np.zeros((8, 8))  # this will keep the real state of the game
    # fake_state = np.zeros((8, 8))  # this will be modified to hold a fake state

    def __init__(self):
        print('Argument List:', str(sys.argv))
        self.play_game(int(sys.argv[2]), sys.argv[1])

    def alpha_beta(self, valid_moves):
        """
        Performs an alpha beta search
        :param valid_moves: the valid moves for current state
        :return: the move that should be taken
        """
        best_move = []
        depth_to_go = 4  # adjust
        alpha = -math.inf  # starting values
        beta = math.inf
        for i in range(len(valid_moves)):
            best_move.append(self.maximize_ab(valid_moves[i], depth_to_go, deepcopy(self.state), self.rnd, alpha, beta))
        return best_move.index(max(best_move))

    def maximize_ab(self, valid_move, depth_to_go, fake_state, rnd, alpha, beta):
        if depth_to_go == 0:  # reached maximum depth
            return self.score_board(fake_state, self.me)  # TODO: Use heuristic here.
        new_fake_state = self.flip_moves(fake_state, valid_move, self.me, self.not_me)
        valid_moves_min = self.get_valid_moves(rnd + 1, self.not_me, new_fake_state)  # get valid moves for hypo state

        move_returned_values = []
        for i in range(len(valid_moves_min)):
            value = self.minimize_ab(valid_moves_min[i], depth_to_go-1, deepcopy(new_fake_state), rnd+1, alpha, beta)
            if value >= beta:
                return value
            alpha = max(value, alpha)
            move_returned_values.append(value)
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amax(move_returned_values)

    def minimize_ab(self, valid_move, depth_to_go, fake_state, rnd, alpha, beta):
        if depth_to_go == 0:
            return self.score_board(fake_state, self.not_me)  # TODO: Use heuristic here.
        new_fake_state = self.flip_moves(fake_state, valid_move, self.not_me, self.me)
        valid_moves_max = self.get_valid_moves(rnd + 1, self.me, new_fake_state)  # get valid moves for hypo state

        move_returned_values = []
        for i in range(len(valid_moves_max)):
            value = (
                self.maximize_ab(valid_moves_max[i], depth_to_go-1, deepcopy(new_fake_state), rnd+1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
            move_returned_values.append(value)
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amin(move_returned_values)

    def min_max(self, valid_moves):
        best_move = []
        depth_to_go = 4
        for i in range(len(valid_moves)):
            best_move.append(self.maximize(valid_moves[i], depth_to_go, deepcopy(self.state), self.rnd))
        return best_move.index(max(best_move))

    def maximize(self, valid_move, depth_to_go, fake_state, rnd):
        if depth_to_go == 0:
            return self.score_board(fake_state, self.me)  # TODO: Use heuristic here.
        new_fake_state = self.flip_moves(fake_state, valid_move, self.me, self.not_me)  # get valid moves for hypo state
        valid_moves_min = self.get_valid_moves(rnd + 1, self.not_me, new_fake_state)

        move_returned_values = []
        for i in range(len(valid_moves_min)):
            move_returned_values.append(
                self.minimize(valid_moves_min[i], depth_to_go-1, deepcopy(new_fake_state), rnd+1))
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amax(move_returned_values)

    def minimize(self, valid_move, depth_to_go, fake_state, rnd):
        if depth_to_go == 0:
            return self.score_board(fake_state, self.not_me)  # TODO: Use heuristic here.
        new_fake_state = self.flip_moves(fake_state, valid_move, self.not_me, self.me)
        valid_moves_max = self.get_valid_moves(rnd + 1, self.me, new_fake_state)  # get valid moves for hypo state

        move_returned_values = []
        for i in range(len(valid_moves_max)):
            move_returned_values.append(
                self.maximize(valid_moves_max[i], (depth_to_go - 1), deepcopy(new_fake_state), rnd + 1))
        if len(move_returned_values) == 0:
            move_returned_values.append(0)
        return np.amin(move_returned_values)

    @staticmethod
    def flip_moves(board, move, me, not_me):
        fake_board = deepcopy(board)
        fake_board[move[0]][move[1]] = me

        # look up the col
        vals_to_potentially_flip = []
        for i in range(move[0], 0, -1):
            if fake_board[i][move[1]] == not_me:
                vals_to_potentially_flip.append([i, move[1]])
            if fake_board[i][move[1]] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[i][move[1]] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[i][move[1]] == 0:
                break

        # look down the col
        for i in range(move[0], len(fake_board[0])):
            if fake_board[i][move[1]] == not_me:
                vals_to_potentially_flip.append([i, move[1]])
            if fake_board[i][move[1]] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[i][move[1]] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[i][move[1]] == 0:
                break

        # look left down the row
        for i in range(move[1], 0, -1):
            if fake_board[move[0]][i] == not_me:
                vals_to_potentially_flip.append([move[0], i])
            if fake_board[move[0]][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[move[0]][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[move[0]][i] == 0:
                break

        # look right down the row
        for i in range(move[1], len(fake_board[0])):
            if fake_board[move[0]][i] == not_me:
                vals_to_potentially_flip.append([move[0], i])
            if fake_board[move[0]][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[move[0]][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[move[0]][i] == 0:
                break

        # look diagonal up
        upval = move[0]
        for i in range(move[1], len(fake_board[0])):
            if upval == -1:
                break
            if fake_board[upval][i] == not_me:
                vals_to_potentially_flip.append([upval, i])
            if fake_board[upval][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == 0:
                break
            upval -= 1

        # look diagonal down
        upval = move[0]
        for i in range(move[1], len(fake_board[0])):
            if upval > 7:
                break
            if fake_board[upval][i] == not_me:
                vals_to_potentially_flip.append([upval, i])
            if fake_board[upval][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == 0:
                break
            upval += 1

        # look diagonal up and back
        upval = move[0]
        for i in range(move[1], 0, -1):
            if upval == -1:
                break
            if fake_board[upval][i] == not_me:
                vals_to_potentially_flip.append([upval, i])
            if fake_board[upval][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == 0:
                break
            upval -= 1

        # look diagonal down and back
        upval = move[0]
        for i in range(move[1], 0, -1):
            if upval > 7:
                break
            if fake_board[upval][i] == not_me:
                vals_to_potentially_flip.append([upval, i])
            if fake_board[upval][i] == 0 and len(vals_to_potentially_flip) > 0:
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == me and len(vals_to_potentially_flip) > 0:
                for j in vals_to_potentially_flip:
                    fake_board[j[0]][j[1]] = me
                vals_to_potentially_flip.clear()
                break
            if fake_board[upval][i] == 0:
                break
            upval += 1

        return fake_board

    @staticmethod
    def score_board(board, me):
        board = np.matrix(board)
        player_one_score = np.count_nonzero(board == 1)
        player_two_score = np.count_nonzero(board == 2)
        if me == 1:
            return np.abs(player_one_score)
        else:
            return np.abs(player_two_score)
        # print("player2: ", player_one_score, "player 1:", player_two_score)

    def move(self, valid_moves):
        """
        Take random corners and sides first.
        :param valid_moves: the valid moves in the current game
        :return: the INDEX within valid_moves
        """
        corner_move = []
        for i in range(0, len(valid_moves)):
            if (valid_moves[i] == self.UPPER_LEFT) or (valid_moves[i] == self.UPPER_RIGHT) or (
                    valid_moves[i] == self.LOWER_LEFT) or (
                    valid_moves[i] == self.LOWER_RIGHT):
                corner_move.append(i)
        if len(corner_move) > 0:
            rand = randint(0, len(corner_move) - 1)
            return corner_move[rand]

        side_move = []
        for i in range(0, len(valid_moves)):
            if (valid_moves[i][0] == 0) or (valid_moves[i][0] == 7) or (valid_moves[i][1] == 0) or (
                    valid_moves[i][1] == 7):
                side_move.append(i)
        if len(side_move) > 0:
            rand = randint(0, len(side_move) - 1)
            return side_move[rand]

        my_move = randint(0, len(valid_moves) - 1)
        return my_move

    def check_dir(self, row, col, incx, incy, me, fake_state=None):
        """
        Figures out if the move at row,col is valid for the given player.
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param incx: ?
        :param incy: ?
        :param me: the given player
        :param fake_state: if changing a hypothetical state
        :return: True or False
        """
        sequence = []
        for i in range(1, 8):
            r = row + incy * i
            c = col + incx * i

            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break
            if fake_state is not None:
                sequence.append(fake_state[r, c])
            else:
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

    def could_be(self, row, col, me, fake_state=None):
        """
        Checks if an unoccupied square can be played by the given player
        :param row: the row of the unoccupied square (int)
        :param col: the col of the unoccupied square (int)
        :param me: the player
        :param fake_state: if changing a hypothetical state
        :return: True or False
        """
        for incx in range(-1, 2):
            for incy in range(-1, 2):
                if (incx == 0) and (incy == 0):  # no need to check curr place
                    continue

                if self.check_dir(row, col, incx, incy, me, fake_state):
                    return True

        return False

    def get_valid_moves(self, rnd, me, fake_state=None):
        """
        Generates the set of valid moves for the player
        :param rnd: what number round the game is currently at.
        :param me: ?
        :param fake_state: if changing a hypothetical state
        :return: A list of valid moves
        """
        valid_moves = []
        if fake_state is not None:
            if rnd < 4:
                if fake_state[3][3] == 0:
                    valid_moves.append([3, 3])
                if fake_state[3][4] == 0:
                    valid_moves.append([3, 4])
                if fake_state[4][3] == 0:
                    valid_moves.append([4, 3])
                if fake_state[4][4] == 0:
                    valid_moves.append([4, 4])
            else:
                for i in range(8):
                    for j in range(8):
                        if fake_state[i][j] == 0:
                            if self.could_be(i, j, me, fake_state):
                                valid_moves.append([i, j])
        else:
            if rnd < 4:
                if self.state[3][3] == 0:
                    valid_moves.append([3, 3])
                if self.state[3][4] == 0:
                    valid_moves.append([3, 4])
                if self.state[4][3] == 0:
                    valid_moves.append([4, 3])
                if self.state[4][4] == 0:
                    valid_moves.append([4, 4])
            else:
                for i in range(8):
                    for j in range(8):
                        if self.state[i][j] == 0:
                            if self.could_be(i, j, me):
                                valid_moves.append([i, j])

        return valid_moves

    def print_game_state(self):
        """
        Uses global variable state to print current game state.
        :return: None
        """
        for i in range(8):  # prints the state of the game in readable rows
            print(self.state[i])

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
        self.rnd = rnd
        return turn, rnd

    def play_game(self, me, host):
        """
        Establishes a connection with the server.
        Then plays whenever it is this player's turn.
        :param me: ?
        :param host: ?
        :return: ?
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
                print("-----------------Round {}-----------------".format(status[1]))  # status[1] has round
                print("My turn.")
                self.print_game_state()

                valid_moves = self.get_valid_moves(status[1], me)  # status[1] has round
                my_move = self.alpha_beta(valid_moves)

                print("Valid moves: {}".format(valid_moves))
                print("Selected move: {}".format(valid_moves[my_move]))

                # Send selection
                selection = str(valid_moves[my_move][0]) + "\n" + str(valid_moves[my_move][1]) + "\n"
                sock.send(selection.encode("utf-8"))
                print("------------------------------------------")
            else:
                print("Other player's turn.")  # Always prints twice?


if __name__ == "__main__":
    """
    Call from command line like this: python AIGuy.py [ip_address] [player_number]
    ip_address is the ip_address on the computer the server was launched on. Enter "localhost" if on same computer
    player_number is 1 (for the black player) and 2 (for the white player)
    """
    # print('Argument List:', str(sys.argv))
    # play_game(int(sys.argv[2]), sys.argv[1])
    AiGuy()
