import requests
import pickle

with open('player_names_list.pkl', 'rb') as file:
    player_names_list = pickle.load(file)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

new_data = []
for i, p in enumerate(player_names_list):
    print(f'getting data for: {i + 1}/{len(player_names_list)}')

    try:
        res_player_stats = requests.get(f'https://api.chess.com/pub/player/{p}/stats', headers = headers).json()
        latest_archive_url = requests.get(f'https://api.chess.com/pub/player/{p}/games/archives', headers = headers).json()['archives'][-1]
        res_latest_month = requests.get(latest_archive_url, headers = headers).json()

        for j, o in enumerate(res_latest_month['games']):
            if p in o['white'].values():
                opponent = o['black']['username']
                acc = o['accuracies']['white']
                opponent_acc = o['accuracies']['black']
            else:
                opponent = o['white']['username']
                acc = o['accuracies']['black']
                opponent_acc = o['accuracies']['white']

            res_opponent_stats = requests.get(f'https://api.chess.com/pub/player/{opponent}/stats', headers = headers).json()

            if o['time_class'] == 'blitz':
                time_class = 0
            elif o['time_class'] == 'rapid':
                time_class = 1
            elif o['time_class'] == 'bullet':
                time_class = 2
            else:
                continue
            
            result_startpos = o['pgn'].find('[Result ') + 9
            result_str = o['pgn'][result_startpos:result_startpos + 3]

            if result_str == '1/2':
                label = 0
            elif result_str == '1-0':
                label = 1
            else:
                label = 2
            
                new_data.append([
                    res_player_stats['chess_blitz']['last']['rating'],
                    res_player_stats['chess_blitz']['best']['rating'],
                    res_player_stats['chess_blitz']['record']['draw'],
                    res_player_stats['chess_blitz']['record']['win'],
                    res_player_stats['chess_blitz']['record']['loss'],
                    res_player_stats['chess_bullet']['last']['rating'],
                    res_player_stats['chess_bullet']['best']['rating'],
                    res_player_stats['chess_bullet']['record']['draw'],
                    res_player_stats['chess_bullet']['record']['win'],
                    res_player_stats['chess_bullet']['record']['loss'],
                    res_player_stats['chess_rapid']['last']['rating'],
                    res_player_stats['chess_rapid']['best']['rating'],
                    res_player_stats['chess_rapid']['record']['draw'],
                    res_player_stats['chess_rapid']['record']['win'],
                    res_player_stats['chess_rapid']['record']['loss'],
                    acc,
                    res_opponent_stats['chess_blitz']['last']['rating'],
                    res_opponent_stats['chess_blitz']['best']['rating'],
                    res_opponent_stats['chess_blitz']['record']['draw'],
                    res_opponent_stats['chess_blitz']['record']['win'],
                    res_opponent_stats['chess_blitz']['record']['loss'],
                    res_opponent_stats['chess_bullet']['last']['rating'],
                    res_opponent_stats['chess_bullet']['best']['rating'],
                    res_opponent_stats['chess_bullet']['record']['draw'],
                    res_opponent_stats['chess_bullet']['record']['win'],
                    res_opponent_stats['chess_bullet']['record']['loss'],
                    res_opponent_stats['chess_rapid']['last']['rating'],
                    res_opponent_stats['chess_rapid']['best']['rating'],
                    res_opponent_stats['chess_rapid']['record']['draw'],
                    res_opponent_stats['chess_rapid']['record']['win'],
                    res_opponent_stats['chess_rapid']['record']['loss'],
                    opponent_acc,
                    time_class,
                    label
                ])
    except:
        continue

with open('gm_wgm_im_wim_new_data.pkl', 'wb') as file:
    pickle.dump(new_data, file)