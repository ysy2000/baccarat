'''
this file is for generating baccarat game results automatically.

output = list of the results of baccarat game among 'player', 'banker', and 'tie'
'''

import time
import os
import random
# import matplotlib
import pandas as pd

num_dec = 8
num_cards_in_one_dec = 52
cut_card = 'cutting card'

num_game = 1

total_cartds = num_dec * num_cards_in_one_dec
result = []

# generate list
def gen_list():

    cards = []
    for i in range(num_dec): 
        for j in range(13):
            # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
            for k in range(4):
                # spade, heart, dia, clova
                if j > 10:  # J, Q, K = 10
                    card_name = "10_" + str(k) + "_" + str(i)
                else:
                    card_name = str(j) + "_" + str(k) + "_" + str(i)
                # ex) 1_1_1 == 1_spade_first dec
                cards.append(card_name)

    if len(cards) == total_cartds:
        print(f'Success! the total number of cards: {total_cartds}')    # 416
    else:
        print(f'Fail to generate right cards')

    return cards
    
    
def compare(player, banker):
    if player > banker:
        return 'player'
    elif banker > player:
        return 'banker'
    else:
        return 'tie'

def get_the_card(suffled_cards, i):
    val = int(suffled_cards[i].split("_")[0])
    if val == cut_card:
        val = int(suffled_cards[i+1].split("_")[0])
        print("The cut card has been scanned. I will change shoes after finishing this game.")
        i += 1
    i += 1

    return val, i

# play game
def play_game(ready_cards):

    # burn cards procedure
    suffled_cards = burn_procedure(ready_cards)

    # value
    results = []

    # draw the cards   
    i = 0

    while i < total_cartds + 1:
        a, i = get_the_card(suffled_cards, i)
        b, i = get_the_card(suffled_cards, i)
        c, i = get_the_card(suffled_cards, i)
        d, i = get_the_card(suffled_cards, i)

        player = (a + c) % 10
        banker = (b + d) % 10

        if player >= 8 or banker >= 8:
            results.append(compare(player, banker))
            break

        elif player > 5 and banker >  5:
            results.append(compare(player, banker))
            break
        
        elif player <= 5:
            p_extra, i = get_the_card(suffled_cards, i)
            player = (player + p_extra) % 10

            if banker <= 2:
                b_extra, i = get_the_card(suffled_cards, i)
                banker = (banker + b_extra) % 10
                results.append(compare(player, banker))
                break
            
            elif banker == 3:
                if p_extra == 8:
                    b_extra, i = get_the_card(suffled_cards, i)
                    banker = (banker + b_extra) % 10
                    results.append(compare(player,banker))
                    break
                results.append(compare(player, banker))
                break

            else:
                # 4 from 2, 5 from 4, 6 from 6 
                if p_extra >= (2 * banker - 6) and p_extra <= 7:
                    b_extra, i = get_the_card(suffled_cards, i)
                    banker = (banker + b_extra) % 10
                    results.append(compare(player,banker))
                    break
                results.append(compare(player, banker))
                break

        elif banker <= 5:
            b_extra, i = get_the_card(suffled_cards, i)
            banker = (banker + b_extra) % 10
            results.append(compare(player,banker))
            break

        else:
            print("calculation error!")
            break
                
    return results

# burn cards procedure
def burn_procedure(suffled_cards):
    burn_num = int(suffled_cards[0].split("_")[0])
    print(f'Burn {burn_num} of cards')
    return suffled_cards[burn_num + 1:]

# suffle
def suffle(cards):
    random.shuffle(cards)

    # insert cut card
    # we usually insert cut card before more than ones dec

    ## ready_cards = suffled_cards[:num_cards_in_one_dec - round( 10 * random())]

    # insert cut card
    cut_point = num_cards_in_one_dec - round( 10 * random.random() )
    cards.insert(cut_point, cut_card)

    return cards

# save
def save_results(result, start):
    file_name = str(start)
    df = pd.DataFrame(result)
    df.to_csv("./data/" + file_name, mode='w', index=False)
    return 0

# analysis the results
def draw_plots():
    return 0

# main
if __name__ == "__main__":
    start = time.time()

    cards_list = gen_list()
    suffled_cards = suffle(cards_list)
    ready_cards = burn_procedure(suffled_cards)
    result = play_game(ready_cards)

    time_sofar = time.time() - start
    Taken_hour = time_sofar//3600
    Taken_min = (time_sofar - Taken_hour*3600)//60
    Time_info = "{}(h) {}(m) Taken".format(Taken_hour, Taken_min )
    print(Time_info)

    save_results(result, start)

    # if you want repeat generation
    for i in range(num_game):
        play_game(suffled_cards)

