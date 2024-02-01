from chessdotcom import get_player_profile, Client, get_leaderboards, get_player_game_archives
import pprint
import requests


Client.request_config["headers"]["User-Agent"] = (
    "My Python Application. "
    "Contact me at conorjackvincent@live.co.uk"
)

printer = pprint.PrettyPrinter()


def print_leaderboards():
    data = get_leaderboards().json
    categories = data.keys()

    

    for category in categories:
        ldrboard = category
    
    for category in data[ldrboard]:
        print('category', category)
        for idx, entry in enumerate(data[ldrboard][category]):
            print(f"Rank: {idx + 1} | Username: {entry['username']} | Rating: {entry['score']}")

def get_most_recent_game(username):
    data = get_player_game_archives(username).json

    url = data['archives'][-1]
    print(url)
    games = requests.get(url)
    print(games)






