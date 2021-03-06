import sys
import socket
import time
from random import randint

GAME_OVER = -999  # the turn is set to -999 if the game is over
UPPER_LEFT = [0, 0]
UPPER_RIGHT = [0, 7]
LOWER_LEFT = [7, 0]
LOWER_RIGHT = [7, 7]

t1 = 0.0  # the amount of time remaining to player 1
t2 = 0.0  # the amount of time remaining to player 2

state = [[0 for x in range(8)] for y in range(8)]  # state[0][0] is the bottom left corner of the board (on the GUI)


def random_move(valid_moves):
    """
    Take a random move
    :param valid_moves: the valid moves in the current game
    :return: the INDEX within valid_moves
    """
    corner_move = []
    for i in range(0, len(valid_moves)):
        if (valid_moves[i] == UPPER_LEFT) or (valid_moves[i] == UPPER_RIGHT) or (valid_moves[i] == LOWER_LEFT) or (
                valid_moves[i] == LOWER_RIGHT):
            corner_move.append(i)
    if len(corner_move) > 0:
        rand = randint(0, len(corner_move) - 1)
        print("PURPOSEFULLY SELECTED A CORNER MOVE")
        return corner_move[rand]

    side_move = []
    for i in range(0, len(valid_moves)):
        if (valid_moves[i][0] == 0) or (valid_moves[i][0] == 7) or (valid_moves[i][1] == 0) or (valid_moves[i][1] == 7):
            side_move.append(i)
    if len(side_move) > 0:
        rand = randint(0, len(side_move) - 1)
        print("PURPOSEFULLY SELECTED A SIDE MOVE")
        return side_move[rand]

    my_move = randint(0, len(valid_moves) - 1)
    print("PICKED A RANDOM MOVE")
    return my_move


def check_dir(row, col, incx, incy, me):
    """
    Figures out if the move at row,col is valid for the given player.
    :param row: the row of the unoccupied square (int)
    :param col: the col of the unoccupied square (int)
    :param incx: ?
    :param incy: ?
    :param me: the given player
    :return: True or False
    """
    sequence = []
    for i in range(1, 8):
        r = row + incy * i
        c = col + incx * i

        if (r < 0) or (r > 7) or (c < 0) or (c > 7):
            break

        sequence.append(state[r][c])

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


def could_be(row, col, me):
    """
    Checks if an unoccupied square can be played by the given player
    :param row: the row of the unoccupied square (int)
    :param col: the col of the unoccupied square (int)
    :param me: the player
    :return: True or False
    """
    for incx in range(-1, 2):
        for incy in range(-1, 2):
            if (incx == 0) and (incy == 0):  # no need to check curr place
                continue

            if check_dir(row, col, incx, incy, me):
                return True

    return False


def get_valid_moves(rnd, me):
    """
    Generates the set of valid moves for the player
    :param rnd: what number round the game is currently at.
    :param me: ?
    :return: A list of valid moves
    """
    valid_moves = []

    if rnd < 4:
        if state[3][3] == 0:
            valid_moves.append([3, 3])
        if state[3][4] == 0:
            valid_moves.append([3, 4])
        if state[4][3] == 0:
            valid_moves.append([4, 3])
        if state[4][4] == 0:
            valid_moves.append([4, 4])
    else:
        for i in range(8):
            for j in range(8):
                if state[i][j] == 0:
                    if could_be(i, j, me):
                        valid_moves.append([i, j])

    return valid_moves


def print_game_state():
    """
    Uses global variable state to print current game state.
    :return: None
    """
    for i in range(8):  # prints the state of the game in readable rows
        print(state[i])


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


def read_message(sock):
    """
    Reads messages from the server that tell the player whose turn it is and what moves are made.
    :param sock: ?
    :return: the current turn and the round
    """
    message = sock.recv(1024).decode("utf-8").split("\n")

    turn = int(message[0])

    if turn == GAME_OVER:
        time.sleep(1)
        sys.exit()

    rnd = int(message[1])

    # update the global state variable
    count = 4
    for i in range(8):
        for j in range(8):
            state[i][j] = int(message[count])
            count += 1

    return turn, rnd


def play_game(me, host):
    """
    Establishes a connection with the server.
    Then plays whenever it is this player's turn.
    :param me: ?
    :param host: ?
    :return: ?
    """
    sock = init_client(me, host)  # Establish connection

    while True:
        status = read_message(sock)  # get update from server

        if status[0] == me:  # status[0] has turn
            print("-----------------Round {}-----------------".format(status[1]))  # status[1] has round
            print("My turn.")
            print_game_state()

            valid_moves = get_valid_moves(status[1], me)  # status[1] has round
            my_move = random_move(valid_moves)  # TODO: Call your move function instead. NOTE: This returns an *index*

            print("Valid moves: {}".format(valid_moves))
            print("Selected move: {}".format(valid_moves[my_move]))

            # Send selection
            selection = str(valid_moves[my_move][0]) + "\n" + str(valid_moves[my_move][1]) + "\n"
            sock.send(selection.encode("utf-8"))
            print("------------------------------------------")
        else:
            print("Other player's turn.")


if __name__ == "__main__":
    """
    Call from command line like this: python AIGuy.py [ip_address] [player_number]
    ip_address is the ip_address on the computer the server was launched on. Enter "localhost" if on same computer
    player_number is 1 (for the black player) and 2 (for the white player)
    """
    print('Argument List:', str(sys.argv))
    play_game(int(sys.argv[2]), sys.argv[1])
