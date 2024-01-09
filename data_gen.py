'''
this file is for generating baccarat game results automatically.

output = list of the results of baccarat game among 'player', 'banker', and 'tie'
'''

import time
import os
from random import random 
import matplotlib

num_dec = 8
num_cards_in_one_dec = 52

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
                card_name = str(j) + "_" + str(k) + "_" + str(i)
                # ex) 1_1_1 == 1_spade_first dec
                cards.append(card_name)

    if len(cards) == total_cartds:
        print(f'Success! the total number of cards: {total_cartds}')    # 416
    else:
        print(f'Fail to generate right cards')

    return cards
    
    

# play game
def play_game(ready_cards):

    # burn cards procedure
    suffled_cards = burn_procedure(ready_cards)

    # value
    results = []
    a, b, c, d = 0

    # draw the cards   
    while i < total_cartds + 1:
        try:
            a = int(suffled_cards[i].split("_")[0])
            b = int(suffled_cards[i+1].split("_")[0])
            c = int(suffled_cards[i+2].split("_")[0])
            d = int(suffled_cards[i+3].split("_")[0])
        except ValueError:
            print("The cut card has been scanned. I will change shoes after finishing this game.")
            ## 컷카드 찾아서 거기만 패스 해서 값을 넣어야함.


        player = (a + c) % 10
        banker = (b + d) % 10

        i += 4

        if player >= 8 or banker >= 8:
            if player > banker:
                results.append('player')
            elif banker > player:
                results.append('banker')
            else:
                results.append('tie')
            break

        elif player > 5 and banker >  5:
            if player > banker:
                results.append('player')
            elif banker > player:
                results.append('banker')
            else:
                results.append('tie')
            break
        
        elif player <= 5:
            e = int(suffled_cards[i].split("_")[0])
            player = (player + e) % 10
            i += 1
            




    

    return 0

# burn cards procedure
def burn_procedure(suffled_cards):
    burn_num = int(suffled_cards[0].split("_")[0])
    print(f'Burn {burn_num} of cards')
    return suffled_cards[burn_num + 1:]

# suffle
def suffle(cards):
    suffled_cards = suffle.cards

    # insert cut card
    # we usually insert cut card before more than ones dec
    ready_cards = suffled_cards[:num_cards_in_one_dec + round( 10 * random() )]
    ## 이 코드가 아니라 컷카드 삽입 코드로 바꿔야함

    return ready_cards

# save
def save_results(list):
    start = time.time()
    time_sofar = time.time() - start
    Taken_hour = time_sofar//3600
    Taken_min = (time_sofar - Taken_hour*3600)//60
    Time_info = "{}(h) {}(m) Taken".format(Taken_hour, Taken_min )
    print(Time_info)
    return 0

# analysis the results
def draw_plots():
    return 0

# main
if __name__ == "__main__":
    cards_list = gen_list()
    suffled_cards = suffle(cards_list)


    for i in range(num_game):
        play_game(suffled_cards)

