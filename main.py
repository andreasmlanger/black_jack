"""
Black Jack Simulation with Card Counting (using multiprocessing)
https://wizardofodds.com/games/blackjack/card-counting/high-low/
"""

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import multiprocessing
import random
import time

# Configuration options
SIMULATIONS = 125000  # per CPU
NUM_DECKS = 4
SHUFFLE_PERCENTAGE = 25
PRINT = False  # print games in console
STRATEGIES = ['Basic Strategy', 'Hi-Lo', 'Hi-Opt I', 'Zen Count']

global stop_signal  # global variable to signal cancel if matplotlib plot is closed

"""
Basic Strategy Tables
"""

#       2  3  4  5  6  7  8  9  10   J   Q   K   A
DECK = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * NUM_DECKS  # multiply with number of decks in game

df_split = pd.DataFrame(columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
df_hard = pd.DataFrame(columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
df_soft = pd.DataFrame(columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#                        2    3    4    5    6    7    8    9   10   11
df_split.loc[11, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y']
df_split.loc[10, :] = ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
df_split.loc[9, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N']
df_split.loc[8, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y']
df_split.loc[7, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N']
df_split.loc[6, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N']
df_split.loc[5, :] = ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
df_split.loc[4, :] = ['N', 'N', 'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N']
df_split.loc[3, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N']
df_split.loc[2, :] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N']
# Y: Split the pair  |  N: Don't split the pair

#                       2    3    4    5    6    7    8    9   10   11
df_hard.loc[17, :] = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
df_hard.loc[16, :] = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'L', 'L', 'L']
df_hard.loc[15, :] = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'L', 'H']
df_hard.loc[14, :] = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
df_hard.loc[13, :] = ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
df_hard.loc[12, :] = ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H']
df_hard.loc[11, :] = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
df_hard.loc[10, :] = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H']
df_hard.loc[9, :] = ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
df_hard.loc[8, :] = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
# H: Hit  |  S: Stand  |  D: Double if allowed, otherwise hit  |  L: Late surrender if allowed, otherwise hit

#                      2    3    4    5    6    7    8    9   10   11
df_soft.loc[9, :] = ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
df_soft.loc[8, :] = ['S', 'S', 'S', 'S', 'd', 'S', 'S', 'S', 'S', 'S']
df_soft.loc[7, :] = ['d', 'd', 'd', 'd', 'd', 'S', 'S', 'H', 'H', 'H']
df_soft.loc[6, :] = ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
df_soft.loc[5, :] = ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
df_soft.loc[4, :] = ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
df_soft.loc[3, :] = ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
df_soft.loc[2, :] = ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H']
# H: Hit  |  S: Stand  |  D: Double if allowed, otherwise hit  |  d: Double if allowed, otherwise stand

"""
Card Counting
"""

ALL_STRATEGIES = ['Basic Strategy', 'Hi-Lo', 'Hi-Opt I', 'Hi-Opt II', 'Omega II', 'Zen Count']
df_counting = pd.DataFrame(index=ALL_STRATEGIES, columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#                         2  3  4  5  6  7  8   9  10  11
df_counting.iloc[0, :] = [0, 0, 0, 0, 0, 0, 0,  0,  0,  0]
df_counting.iloc[1, :] = [1, 1, 1, 1, 1, 0, 0,  0, -1, -1]
df_counting.iloc[2, :] = [0, 1, 1, 1, 1, 0, 0,  0, -1,  0]
df_counting.iloc[3, :] = [1, 1, 2, 2, 1, 1, 0,  0, -2,  0]
df_counting.iloc[4, :] = [1, 1, 2, 2, 2, 1, 0, -1, -2,  0]
df_counting.iloc[5, :] = [1, 1, 2, 2, 2, 1, 0,  0, -2, -1]


def card_counter(cards, strategy='Basic Strategy'):
    return sum([df_counting.loc[strategy][card] for card in cards])


def true_count(deck, r_cnt):
    if r_cnt >= 0:
        return r_cnt // (max(52, len(deck)) // 52)
    return -(-r_cnt // (max(52, len(deck)) // 52))


def simulate(queue, strategy):
    deck = []

    def new_deck():
        d = DECK[:]
        random.shuffle(d)
        return d

    def deal_cards(count):
        player_hands = [[deck.pop()]]  # all player cards initially in one hand
        dealer_cards = [deck.pop()]
        player_hands[0].append(deck.pop())
        dealer_cards.append(deck.pop())

        count += card_counter(player_hands[0] + dealer_cards[:1], strategy)

        if PRINT:
            print(f'Player Cards: {player_hands[0]} | Dealer Cards: [', end='')
            print(str(dealer_cards[0]) + ', \033[31m' + str(dealer_cards[1]) + '\033[0m]')

        return player_hands, dealer_cards, count

    def split_cards(player_hands, dealer_cards, count):
        if player_hands[0][0] == player_hands[0][1]:  # check if cards can be split
            check = df_split.loc[player_hands[0][0]][dealer_cards[0]]

            if strategy != 'Basic Strategy' and player_hands[0] == 10:  # deviations according to 'Illustrious 18'
                if dealer_cards[0] == 5 and true_count(deck, count) >= 5:
                    check = 'Y'
                elif dealer_cards[0] == 6 and true_count(deck, count) >= 4:
                    check = 'Y'

            if check == 'Y':
                for _ in range(2):  # split a max of 3 times
                    hands = []
                    for hand in player_hands:
                        if hand[0] == hand[1]:
                            for j in range(2):
                                hands.append([hand[j]])
                                hands[-1].append(deck.pop())
                                count += card_counter([hands[-1][-1]], strategy)
                        else:
                            hands.append(hand)
                    player_hands = hands

        if PRINT and len(player_hands) > 1:
            print('\033[32m' + 'Split into ' + str(len(player_hands)) + ' hands\033[0m')
            for j in range(len(player_hands)):
                print(f'Player Cards: {player_hands[j]} | Dealer Cards: [', end='')
                print(str(dealer_cards[0]) + ', \033[31m' + str(dealer_cards[1]) + '\033[0m]')

        return player_hands, count

    def hit(cards, count, acc):
        cards.append(deck.pop())
        count += card_counter(cards[-1:], strategy)
        acc += cards[-1]
        if PRINT:
            print('\033[36m' + 'Hit' + '\033[0m')
        return cards, count, acc

    def double_down():
        if PRINT:
            print('\033[32m' + 'Double Down' + '\033[0m')
        return True

    def deviate(count, bet):
        if PRINT:
            print('\033[32m' + 'Deviation (', end='')
            print(f'running count: {count} | true count: {true_count(deck, count)} | bet: {bet})', end='')
            print('\033[0m')

    def place_bet(tr_count):
        max_bet = 50
        return max(1, min(tr_count, max_bet))  # bet more if true count is high

    def play_hand(r_cnt):
        bet = place_bet(true_count(deck, r_cnt))

        results = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # Black Jack, win, tie, lose, surrender, insurance
        double = [False, False, False, False]  # max of 4 hands is played
        surrender = [False, False, False, False]
        insurance = False
        player_hands, dealer_cards, r_cnt = deal_cards(r_cnt)

        # Insurance according to 'Illustrious 18'
        if dealer_cards[0] == 11 and true_count(deck, r_cnt) >= 3:
            if PRINT:
                print('\033[32m' + 'Insurance' + '\033[0m')
            insurance = True
            results[5][0] += 1

        d_sum = sum(dealer_cards)

        if d_sum == 21:  # dealer peeks for natural Black Jack
            if insurance:
                results[5][1] += bet  # won insurance bet
            if PRINT:
                print('Dealer wins - BLACKJACK')
            results[3][0] = 1
            results[3][1] = -bet

        else:
            if insurance:
                results[5][1] -= 0.5 * bet  # lost insurance bet

            # Basic strategy splitting
            player_hands, r_cnt = split_cards(player_hands, dealer_cards, r_cnt)

            for j, player_cards in enumerate(player_hands):

                p_sum = sum(player_cards)

                while True:
                    if double[j]:  # check if double, then only 1 more card is allowed
                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                        if p_sum > 21 and 11 in player_cards:  # convert Ace to 1
                            player_cards[player_cards.index(11)] = 1
                            p_sum -= 10
                        break

                    if p_sum > 21 and 11 in player_cards:  # convert Ace to 1
                        player_cards[player_cards.index(11)] = 1
                        p_sum -= 10
                    elif p_sum > 21:  # Bust
                        break

                    # Basic strategy soft totals
                    if 11 in player_cards and 12 < p_sum < 21:
                        check = df_soft.loc[p_sum - 11][dealer_cards[0]]

                        if len(player_cards) == 2 and (check == 'D' or check == 'd'):
                            double[j] = double_down()
                            continue
                        elif check == 'H' or check == 'D':
                            player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                            continue
                        elif check == 'S' or check == 'd':
                            break

                    # Basic strategy hard totals
                    if p_sum < 8:
                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                        continue
                    elif 7 < p_sum < 18:

                        # Deviations from basic strategy according to 'Illustrious 18'
                        if strategy != 'Basic Strategy':
                            if p_sum == 16:
                                if dealer_cards[0] == 10:
                                    if r_cnt >= 0:
                                        deviate(r_cnt, bet)
                                        break
                                    elif true_count(deck, r_cnt) >= -2 and len(player_cards) == 2:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                                    else:
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 9:
                                    if true_count(deck, r_cnt) >= 5:
                                        deviate(r_cnt, bet)
                                        break
                                    elif r_cnt >= 0 and len(player_cards) == 2:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                                    else:
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 11:
                                    if true_count(deck, r_cnt) >= -4 and len(player_cards) == 2:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                                    else:
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                            elif p_sum == 15:
                                if dealer_cards[0] == 10:
                                    if true_count(deck, r_cnt) >= 4:
                                        deviate(r_cnt, bet)
                                        break
                                    elif r_cnt > 0 and len(player_cards) == 2:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                                    else:
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 9:
                                    if true_count(deck, r_cnt) >= 2:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                                elif dealer_cards[0] == 11:
                                    if true_count(deck, r_cnt) >= 1:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                            elif p_sum == 14:
                                if dealer_cards[0] == 10:
                                    if true_count(deck, r_cnt) >= 3:
                                        deviate(r_cnt, bet)
                                        surrender[j] = True
                                        break
                            elif p_sum == 13:
                                if dealer_cards[0] == 2:
                                    if r_cnt < 0:
                                        deviate(r_cnt, bet)
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 3:
                                    if true_count(deck, r_cnt) < -2:
                                        deviate(r_cnt, bet)
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                            elif p_sum == 12:
                                if dealer_cards[0] == 2:
                                    if true_count(deck, r_cnt) >= 3:
                                        deviate(r_cnt, bet)
                                        break
                                elif dealer_cards[0] == 3:
                                    if true_count(deck, r_cnt) >= 2:
                                        deviate(r_cnt, bet)
                                        break
                                elif dealer_cards[0] == 4:
                                    if r_cnt < 0:
                                        deviate(r_cnt, bet)
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 5:
                                    if true_count(deck, r_cnt) < -2:
                                        deviate(r_cnt, bet)
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                                elif dealer_cards[0] == 6:
                                    if true_count(deck, r_cnt) < -1:
                                        deviate(r_cnt, bet)
                                        player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                        continue
                            if len(player_cards) == 2:
                                if p_sum == 11:
                                    if dealer_cards[0] == 11:
                                        if true_count(deck, r_cnt) < 1:
                                            deviate(r_cnt, bet)
                                            player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                                            continue
                                elif p_sum == 10:
                                    if dealer_cards[0] == 11 or dealer_cards[0] == 10:
                                        if true_count(deck, r_cnt) >= 4:
                                            deviate(r_cnt, bet)
                                            double[j] = double_down()
                                            continue
                                elif p_sum == 9:
                                    if dealer_cards[0] == 7:
                                        if true_count(deck, r_cnt) >= 3:
                                            deviate(r_cnt, bet)
                                            double[j] = double_down()
                                            continue
                                    elif dealer_cards[0] == 2:
                                        if true_count(deck, r_cnt) >= 1:
                                            deviate(r_cnt, bet)
                                            double[j] = double_down()
                                            continue

                        check = df_hard.loc[p_sum][dealer_cards[0]]  # if no deviations, follow basic strategy

                        if len(player_cards) == 2 and check == 'D':
                            double[j] = double_down()
                            continue
                        elif len(player_cards) == 2 and check == 'L':
                            surrender[j] = True
                            break
                        elif check == 'S':
                            break
                        else:
                            player_cards, r_cnt, p_sum = hit(player_cards, r_cnt, p_sum)
                            continue
                    else:  # sum > 17
                        break

                player_hands[j] = player_cards

            if PRINT and (len(player_hands[0]) > 2 or (len(player_hands) > 1 and len(player_hands[1]) > 2)):
                print(f'Player Cards: {player_hands[0]} | Dealer Cards {dealer_cards}')
                if len(player_hands) > 1:
                    print(f'Player Cards: {player_hands[1]} | Dealer Cards {dealer_cards}')

            # Dealer strategy
            r_cnt += card_counter(dealer_cards[-1:], strategy)  # second card of dealer
            if min([sum(cards) for cards in player_hands if cards is not None]) < 22 and not all(surrender):
                while True:
                    while d_sum < 17:
                        dealer_cards.append(deck.pop())
                        r_cnt += card_counter(dealer_cards[-1:], strategy)
                        d_sum = d_sum + dealer_cards[-1]
                        if PRINT:
                            print('\033[35m' + 'Dealer Hit' + '\033[0m')

                    if d_sum > 21 and 11 in dealer_cards:
                        dealer_cards[dealer_cards.index(11)] = 1
                        d_sum -= 10
                    else:
                        break

            for j, cards in enumerate(player_hands):
                p_sum = sum(cards)

                if PRINT and len(dealer_cards) > 2:
                    print(f'Player Cards: {cards} | Dealer Cards: {dealer_cards}')

                if surrender[j]:
                    if PRINT:
                        print('Player surrenders')
                    results[4][0] += 1
                    results[4][1] -= 0.5 * bet
                elif p_sum == 21 and len(cards) == 2 and d_sum != 21:
                    if PRINT:
                        print('Player wins - BLACKJACK')
                    results[0][0] += 1
                    results[0][1] += 1.5 * bet * (1 + double[j])
                elif p_sum > 21:
                    if PRINT:
                        print('Dealer wins - Player bust')
                    results[3][0] += 1
                    results[3][1] -= bet * (1 + double[j])
                elif d_sum > 21:
                    if PRINT:
                        print('Player wins - Dealer bust')
                    results[1][0] += 1
                    results[1][1] += bet * (1 + double[j])
                elif d_sum == p_sum:
                    if PRINT:
                        print('Tie')
                    results[2][0] += 1
                elif d_sum > p_sum:
                    if PRINT:
                        print('Dealer wins')
                    results[3][0] += 1
                    results[3][1] -= bet * (1 + double[j])
                elif d_sum < p_sum:
                    if PRINT:
                        print('Player wins')
                    results[1][0] += 1
                    results[1][1] += bet * (1 + double[j])

        queue.put((results, true_count(deck, r_cnt)))

        if PRINT:
            print('--------------------')

        return r_cnt

    deck = new_deck()
    r_count = 0

    for i in range(SIMULATIONS):
        if len(deck) / 52 / NUM_DECKS * 100 < SHUFFLE_PERCENTAGE:
            deck = new_deck()  # Reshuffle
            r_count = 0
            if PRINT:
                print('\033[94m' + 'Shuffle' + '\033[0m')

        # Play hand
        r_count = play_hand(r_count)


def main():
    def handle_close(_):
        global stop_signal
        stop_signal = True

        for p in processes:
            p.terminate()

    cpus = 1 if PRINT else multiprocessing.cpu_count()  # prevent concurrency issues when printing

    fig = plt.figure(figsize=(9, 4), dpi=120)
    fig.canvas.mpl_connect('close_event', handle_close)

    ax1 = plt.subplot2grid((10, 10), (0, 0), colspan=6, rowspan=10)
    ax1.set_xlabel('Black Jack Games Played')
    ax1.set_ylabel('Total Balance')

    ax3 = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=3)
    ax3.set_xlabel('')
    ax3.set_xlim(left=-20, right=20)
    ax3.set_ylabel('')
    ax3.set_title('True Count', fontsize=10)

    ax2 = plt.subplot2grid((10, 10), (4, 7), colspan=3, rowspan=6)
    ax2.set_xlabel('')
    ax2.set_ylabel('Edge over House (%)')

    plt.axhline(linestyle='--', linewidth=0.5, color='black')

    labels = [label.replace(' ', '\n') for label in STRATEGIES]  # text wrap
    edge = [0.0 for _ in STRATEGIES]

    cm_colors = iter(colormaps['magma'](np.linspace(0, 0.7, len(STRATEGIES))))
    colors = [next(cm_colors) for _ in range(len(STRATEGIES))]

    for i, strategy in enumerate(STRATEGIES):
        global stop_signal
        stop_signal = False
        random.seed(42)

        start_time = time.time()

        queue = multiprocessing.Queue()
        processes = []

        for _ in range(cpus):
            process = multiprocessing.Process(target=simulate, args=(queue, strategy))
            processes.append(process)
            process.start()

        results = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])  # stores results of simulation
        balance = [0.0] * (SIMULATIONS * cpus + 1)
        game = 0
        interval = max(1, SIMULATIONS * cpus // 1000)
        bar, hist = None, None

        ax1.plot(0, 0, '-', color=colors[i], label=strategy)
        ax1.legend(loc='upper left')

        formatter = EngFormatter(sep='')
        ax1.xaxis.set_major_formatter(formatter)
        ax1.yaxis.set_major_formatter(formatter)

        counts = dict()

        plt.draw()
        plt.pause(1)

        while True:
            if not queue.empty():
                game += 1
                package = queue.get(block=False)
                results = np.add(results, package[0])
                if package[1] not in counts:
                    counts[package[1]] = 0
                counts[package[1]] += 1
                balance[game] = np.sum(results, axis=0, dtype=float)[1]

                if (game % interval == 0 or game == SIMULATIONS * cpus) and not stop_signal:  # update plot

                    # ax1
                    x = range(max(0, game - interval - 1), game + 1)
                    y = balance[max(0, game - interval - 1):game + 1]
                    ax1.plot(x, y, '-', color=colors[i])
                    ax1.autoscale(True)
                    ax1.relim()

                    # ax2
                    if bar:
                        bar.remove()  # Update bars
                    edge[i] = balance[game] / game * 100
                    bar = ax2.bar(labels, edge, color=colors)
                    ax2.relim()
                    ax2.autoscale(True)

                    # ax3
                    if strategy != 'Basic Strategy':  # otherwise, true count is always zero
                        if hist:
                            hist.remove()
                        x_hist, y_hist = list(counts.keys()), [y / game for y in list(counts.values())]
                        hist = ax3.bar(x_hist, y_hist, color=colors[i], alpha=0.5)
                        ax3.relim()
                        ax3.autoscale(True)
                        ax3.set_xlim(left=-20, right=20)

                    plt.draw()
                    plt.pause(0.1)

            if all([process.exitcode is not None for process in processes]) & queue.empty():
                break

        finish_time = time.time() - start_time

        print('\n\033[1m[' + strategy.upper() + ']\033[0m')
        print('Cores: %d' % cpus, end=' | ')
        print('Total simulations: %d' % game, end=' | ')
        print('Simulations/s: %d' % (game / finish_time), end=' | ')
        print('Execution time: %.2fs' % finish_time)
        print('Black Jack percentage: %.2f%%' % float(results[0][0] / game * 100))
        print('Win percentage: %.2f%%' % float(results[1][0] / game * 100))
        print('Tie percentage: %.2f%%' % float(results[2][0] / game * 100))
        print('Lose percentage: %.2f%%' % float(results[3][0] / game * 100))
        print('Surrender percentage: %.2f%%' % float(results[4][0] / game * 100))

        if edge[i] > 0:
            print('Edge over house:\033[36m %.2f%%\033[0m' % edge[i])
        else:
            print('Edge over house:\033[31m %.2f%%\033[0m' % edge[i])

        if stop_signal:
            break

    plt.savefig('simulation.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
